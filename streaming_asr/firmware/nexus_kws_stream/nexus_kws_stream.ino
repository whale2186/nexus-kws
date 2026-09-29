#include <Arduino.h>
#include <WiFi.h>
#include <driver/i2s.h>
#include "esp_dsp.h"
#include "model.h"
#include "mfcc_coeffs.h"
#include "mfcc_norm.h"

// TensorFlow Lite Micro
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// --- WI-FI & ASR SERVER CONFIG ---
#if __has_include("secrets.h")
  #include "secrets.h"
#endif

#ifndef WIFI_SSID
  #define WIFI_SSID       "YOUR_WIFI_SSID"
  #define WIFI_PASS       "YOUR_WIFI_PASSWORD"
#endif

#ifndef ASR_SERVER_IP
  #define ASR_SERVER_IP   "192.168.1.100"
  #define ASR_SERVER_PORT 5000
#endif

// --- HARDWARE CONFIGURATION ---
#define I2S_PORT        I2S_NUM_0
#define PIN_MIC_SD      32
#define PIN_MIC_SCK     18
#define PIN_MIC_WS      19

#define SAMPLE_RATE     16000
#define N_FFT           512
#define HOP_SAMPLES     320
#define NUM_FRAMES      49
#define N_MELS          40
#define N_MFCC          13

#define DIGITAL_GAIN    16.0f
#define VAD_THRESHOLD   2600 // Gate ambient noise w/ WiFi RF (~1800-2400 RMS), speech is ~6000-16000 RMS
#define DC_R            0.995f

// --- PRE-ROLL BUFFER (200ms = 10 hops * 320 samples = 3,200 samples = 6.4KB) ---
#define PREROLL_HOPS    10
#define PREROLL_SAMPLES (PREROLL_HOPS * HOP_SAMPLES)
int16_t preroll_buffer[PREROLL_SAMPLES];
volatile int preroll_head = 0;

// Streaming state
volatile bool is_streaming = false;
QueueHandle_t audio_stream_queue = nullptr;

// --- MFCC PIPELINE VARIABLES ---
float audio_frame[N_FFT];
float hann_window[N_FFT];
float fft_buffer[2 * N_FFT];
float power_spectrum[N_FFT / 2 + 1];
float mel_energies[N_MELS];
float mfcc_matrix[N_MFCC][NUM_FRAMES];

portMUX_TYPE matrix_mux = portMUX_INITIALIZER_UNLOCKED;
portMUX_TYPE stream_mux = portMUX_INITIALIZER_UNLOCKED;

// --- TFLM CONFIG ---
constexpr int kTensorArenaSize = 24 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// DSP & VAD state
static float dc_x1 = 0.0f, dc_y1 = 0.0f;
volatile float latest_rms = 0.0f;
volatile int warmup_frames = 0;
#define WARMUP_REQUIRED 49
volatile int new_frames_since_inference = 0;

// KWS state machine
uint32_t cooldown_until = 0;
int trigger_streak = 0;

// CPU profiling
volatile uint32_t core0_active_us = 0;
volatile uint32_t core1_active_us = 0;
uint32_t last_cpu_report_ms = 0;

void setup_i2s() {
    i2s_config_t cfg = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 320,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pins = {
        .bck_io_num = PIN_MIC_SCK,
        .ws_io_num = PIN_MIC_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = PIN_MIC_SD
    };

    i2s_driver_install(I2S_PORT, &cfg, 0, NULL);
    i2s_set_pin(I2S_PORT, &pins);
    i2s_zero_dma_buffer(I2S_PORT);

    // Flush start transients
    int32_t trash[320];
    size_t bytes_read;
    for (int i = 0; i < 150; i++) {
        i2s_read(I2S_PORT, trash, sizeof(trash), &bytes_read, portMAX_DELAY);
    }
}

void setup_tflite() {
    model = tflite::GetModel(model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema mismatch!");
        while (1);
    }

    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddMean();
    resolver.AddAdd();
    resolver.AddQuantize();
    resolver.AddDequantize();

    static tflite::MicroErrorReporter micro_error_reporter;
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter
    );
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("AllocateTensors failed!");
        while (1);
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
}

void compute_mfcc(int16_t* new_samples, int hop_size) {
    int hop = 320;
    int win = 512;
    for (int i = 0; i < win - hop; i++) {
        audio_frame[i] = audio_frame[i + hop];
    }
    for (int i = 0; i < hop; i++) {
        audio_frame[win - hop + i] = (float)new_samples[i] / 32768.0f;
    }

    for (int i = 0; i < win; i++) {
        fft_buffer[2 * i] = audio_frame[i] * hann_window[i];
        fft_buffer[2 * i + 1] = 0.0f;
    }

    dsps_fft2r_fc32(fft_buffer, N_FFT);
    dsps_bit_rev2r_fc32(fft_buffer, N_FFT);

    for (int k = 0; k < (N_FFT / 2 + 1); k++) {
        float r = fft_buffer[2 * k];
        float im = fft_buffer[2 * k + 1];
        power_spectrum[k] = r * r + im * im;
    }

    for (int m = 0; m < N_MELS; m++) {
        float energy = 0.0f;
        for (int k = 0; k < (N_FFT / 2 + 1); k++) {
            float w = mel_basis[m][k];
            if (w > 0.0f) {
                energy += w * power_spectrum[k];
            }
        }
        if (energy < 1e-10f) energy = 1e-10f;
        mel_energies[m] = 10.0f * log10f(energy);
    }

    float current_frame_mfcc[N_MFCC];
    for (int c = 0; c < N_MFCC; c++) {
        float sum = 0.0f;
        for (int m = 0; m < N_MELS; m++) {
            sum += dct_basis[c][m] * mel_energies[m];
        }
        current_frame_mfcc[c] = (sum - mfcc_mean[c]) / mfcc_std[c];
    }

    portENTER_CRITICAL(&matrix_mux);
    for (int c = 0; c < N_MFCC; c++) {
        for (int f = 0; f < NUM_FRAMES - 1; f++) {
            mfcc_matrix[c][f] = mfcc_matrix[c][f + 1];
        }
        mfcc_matrix[c][NUM_FRAMES - 1] = current_frame_mfcc[c];
    }
    if (warmup_frames < WARMUP_REQUIRED) warmup_frames++;
    new_frames_since_inference++;
    portEXIT_CRITICAL(&matrix_mux);
}

// Audio Task pinned to Core 0 (Continuous I2S Capture + DSP + Pre-roll Ring Buffer)
void audio_task(void* param) {
    int32_t raw32[320 * 2];
    int16_t pcm[320];
    size_t bytesRead = 0;

    while (1) {
        esp_err_t err = i2s_read(I2S_PORT, raw32, sizeof(raw32), &bytesRead, portMAX_DELAY);
        if (err == ESP_OK && bytesRead == sizeof(raw32)) {
            uint32_t t_c0 = micros();

            float rms = 0;
            for (int i = 0; i < 320; i++) {
                int32_t raw24 = raw32[i * 2];
                int16_t s16 = (int16_t)(raw24 >> 14);
                float x = (float)s16;
                float y = x - dc_x1 + DC_R * dc_y1;
                dc_x1 = x; dc_y1 = y;

                float amplified = y * DIGITAL_GAIN;
                rms += amplified * amplified;

                if (amplified > 32767) amplified = 32767;
                if (amplified < -32768) amplified = -32768;
                pcm[i] = (int16_t)amplified;
            }
            latest_rms = sqrtf(rms / 320.0f);

            // Store in pre-roll circular buffer
            for (int i = 0; i < 320; i++) {
                preroll_buffer[(preroll_head + i) % PREROLL_SAMPLES] = pcm[i];
            }
            preroll_head = (preroll_head + 320) % PREROLL_SAMPLES;

            // If active streaming to ASR, transmit PCM chunk directly over FreeRTOS queue
            if (is_streaming && audio_stream_queue != nullptr) {
                xQueueSend(audio_stream_queue, pcm, 0);
            }

            compute_mfcc(pcm, 320);
            core0_active_us += (micros() - t_c0);
        } else {
            vTaskDelay(pdMS_TO_TICKS(5));
        }
    }
}

void stream_to_asr() {
    uint32_t t_wake = millis();
    Serial.printf("\n[STREAM] Triggering ASR Handoff to %s:%d ...\n", ASR_SERVER_IP, ASR_SERVER_PORT);

    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[STREAM ERROR] Wi-Fi not connected!");
        return;
    }

    WiFiClient client;
    if (!client.connect(ASR_SERVER_IP, ASR_SERVER_PORT, 1000)) {
        Serial.println("[STREAM ERROR] Failed to connect to ASR server!");
        return;
    }

    uint32_t t_connected = millis();
    Serial.printf("[STREAM] TCP Connected in %d ms! Sending pre-roll buffer (%d ms)...\n",
                  t_connected - t_wake, (PREROLL_SAMPLES * 1000) / SAMPLE_RATE);

    // 1. Dump pre-roll buffer in chronological order
    int head = preroll_head;
    int16_t temp_buf[320];
    for (int h = 0; h < PREROLL_HOPS; h++) {
        int start_idx = (head + h * 320) % PREROLL_SAMPLES;
        for (int i = 0; i < 320; i++) {
            temp_buf[i] = preroll_buffer[(start_idx + i) % PREROLL_SAMPLES];
        }
        client.write((const uint8_t*)temp_buf, 320 * sizeof(int16_t));
    }

    if (audio_stream_queue != nullptr) {
        xQueueReset(audio_stream_queue);
    }
    is_streaming = true;

    Serial.println("[STREAM] Pre-roll sent. Live streaming command audio...");

    // 2. Stream live audio from queue until silence detected for >1.2s or max 5.0s
    uint32_t stream_start = millis();
    int silence_hops = 0;
    int16_t live_pcm[320];

    while (client.connected()) {
        if (xQueueReceive(audio_stream_queue, live_pcm, pdMS_TO_TICKS(50)) == pdTRUE) {
            client.write((const uint8_t*)live_pcm, 320 * sizeof(int16_t));

            if (latest_rms < VAD_THRESHOLD) {
                silence_hops++;
            } else {
                silence_hops = 0;
            }

            if (silence_hops >= 60 || (millis() - stream_start >= 5000)) {
                Serial.printf("[STREAM] Speech finished (silence=%d hops, elapsed=%d ms). Closing stream.\n",
                              silence_hops, millis() - stream_start);
                break;
            }
        } else {
            if (millis() - stream_start >= 5000) break;
        }
    }

    is_streaming = false;
    client.flush();
    client.stop();

    uint32_t total_stream_duration = millis() - stream_start;
    Serial.printf("[STREAM COMPLETE] Streamed %d ms of command audio to ASR server.\n\n",
                  total_stream_duration);

    cooldown_until = millis() + 1000;
}

void setup() {
    Serial.setTxBufferSize(4096);
    Serial.setRxBufferSize(1024);
    Serial.begin(921600);
    delay(500);
    Serial.println("\n========================================================");
    Serial.println("Nexus KWS + Real-Time ASR Streaming Handoff");
    Serial.println("========================================================");

    // Initialize Wi-Fi in Station mode (connects in background without stalling audio)
    Serial.printf("Connecting to Wi-Fi SSID: %s ...\n", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(WIFI_PS_MIN_MODEM);
    WiFi.setTxPower(WIFI_POWER_8_5dBm);
    WiFi.begin(WIFI_SSID, WIFI_PASS);

    setup_tflite();
    setup_i2s();

    audio_stream_queue = xQueueCreate(20, 320 * sizeof(int16_t));

    esp_err_t dsp_err = dsps_fft2r_init_fc32(NULL, N_FFT);
    if (dsp_err != ESP_OK) {
        Serial.printf("ESP-DSP FFT init failed: %d\n", dsp_err);
    }
    for (int i = 0; i < N_FFT; i++) {
        hann_window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (float)N_FFT));
    }

    // Start Audio Acquisition Task on Core 0
    xTaskCreatePinnedToCore(
        audio_task,
        "AudioEngine",
        8192,
        NULL,
        2,
        NULL,
        0
    );

    Serial.println("Initialization Complete. Listening for 'Nexus'...\n");
}

void loop() {
    // Wi-Fi status monitor (non-blocking)
    static bool wifi_connected_logged = false;
    if (WiFi.status() == WL_CONNECTED && !wifi_connected_logged) {
        wifi_connected_logged = true;
        Serial.printf("[Wi-Fi] Connected! Local IP: %s | RSSI: %d dBm\n",
                      WiFi.localIP().toString().c_str(), WiFi.RSSI());
    } else if (WiFi.status() != WL_CONNECTED && wifi_connected_logged) {
        wifi_connected_logged = false;
        Serial.println("[Wi-Fi] Disconnected, attempting reconnect...");
    }

    // Periodic CPU telemetry (reported every second)
    if (millis() - last_cpu_report_ms >= 1000) {
        uint32_t now = millis();
        uint32_t elapsed_us = (now - last_cpu_report_ms) * 1000;
        last_cpu_report_ms = now;

        float cpu0 = (core0_active_us * 100.0f) / elapsed_us;
        float cpu1 = (core1_active_us * 100.0f) / elapsed_us;
        core0_active_us = 0;
        core1_active_us = 0;
        if (cpu0 > 100.0f) cpu0 = 100.0f;
        if (cpu1 > 100.0f) cpu1 = 100.0f;
        float total_cpu = (cpu0 + cpu1) / 2.0f;

        Serial.printf("[CPU REPORT] Core 0 (Audio): %.1f%% | Core 1 (ML/App): %.1f%% | Combined: %.1f%%\n",
                      cpu0, cpu1, total_cpu);
    }

    // Cooldown check
    if (millis() < cooldown_until) {
        delay(10);
        return;
    }

    // Warmup check
    if (warmup_frames < WARMUP_REQUIRED) {
        new_frames_since_inference = 0;
        delay(10);
        return;
    }

    // VAD Gate
    float rms = latest_rms;
    if (rms < VAD_THRESHOLD) {
        trigger_streak = 0;
        static uint32_t last_idle_print_ms = 0;
        if (millis() - last_idle_print_ms >= 1000) {
            last_idle_print_ms = millis();
            Serial.printf("[IDLE] RMS: %.0f (VAD Gate: %d)\n", rms, VAD_THRESHOLD);
        }
        delay(20);
        return;
    }

    // Only run inference if at least 2 new audio hops (40ms) entered
    if (new_frames_since_inference < 2) {
        delay(5);
        return;
    }

    uint32_t t_c1 = micros();

    // Prepare INT8 quantized tensor
    float input_scale = input_tensor->params.scale;
    int32_t input_zero_point = input_tensor->params.zero_point;
    int8_t* dst = input_tensor->data.int8;

    portENTER_CRITICAL(&matrix_mux);
    for (int m = 0; m < N_MFCC; m++) {
        for (int f = 0; f < NUM_FRAMES; f++) {
            float val = mfcc_matrix[m][f];
            int32_t q = (int32_t)roundf(val / input_scale) + input_zero_point;
            if (q < -128) q = -128;
            if (q > 127)  q = 127;
            dst[m * NUM_FRAMES + f] = (int8_t)q;
        }
    }
    new_frames_since_inference = 0;
    portEXIT_CRITICAL(&matrix_mux);

    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Invoke failed!");
        return;
    }

    // Check Wake Word Confidence (index 1 = "Nexus", index 0 = negative/silence)
    int8_t wakeword_score = output_tensor->data.int8[1];
    float out_scale = output_tensor->params.scale;
    int32_t out_zero_point = output_tensor->params.zero_point;
    float prob = ((int)wakeword_score - (int)out_zero_point) * out_scale;
    if (prob < 0.0f) prob = 0.0f;
    if (prob > 1.0f) prob = 1.0f;

    core1_active_us += (micros() - t_c1);

    Serial.printf("Prob: %.2f | Streak: %d | RMS: %.0f\n", prob, trigger_streak, rms);

    // Wake word decision logic
    bool triggered = false;
    if (prob >= 0.88f) {
        triggered = true;
    } else if (prob >= 0.72f) {
        trigger_streak++;
        if (trigger_streak >= 2) {
            triggered = true;
        }
    } else {
        trigger_streak = 0;
    }

    if (triggered) {
        cooldown_until = millis() + 1500;
        new_frames_since_inference = 0;
        trigger_streak = 0;

        Serial.println("\n***************************************************");
        Serial.printf(">>>> WAKE WORD DETECTED! Prob: %.2f <<<<\n", prob);
        Serial.println("***************************************************");

        // Perform streaming handoff to remote ASR server
        stream_to_asr();
    }
}
