#include <Arduino.h>
#include <WiFi.h>
#include <driver/i2s.h>
#include "esp_dsp.h"
#include "model.h"
#include "mfcc_coeffs.h"
#include "mfcc_norm.h"

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// --- AUDIO CONFIG ---
#define SAMPLE_RATE 16000
#define I2S_PORT I2S_NUM_0
#define PIN_MIC_SD 32
#define PIN_MIC_SCK 18
#define PIN_MIC_WS 19
#define NUM_FRAMES 49 
#define DIGITAL_GAIN 16.0f
#define VAD_THRESHOLD 1800 // Gate ambient noise (~600-1400 RMS), speech is ~3500-25000 RMS
#define HOP_SAMPLES   320

// --- TFLITE CONFIG ---
constexpr int kTensorArenaSize = 48 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// --- MFCC PIPELINE ---
float audio_frame[N_FFT];     // 512 points for FFT
float hann_window[N_FFT];     // Precomputed periodic Hann window
float fft_buffer[2 * N_FFT];  // Interleaved complex buffer [Re0, Im0, Re1, Im1, ...]
float power_spectrum[N_FFT / 2 + 1]; // Power spectrum (|X[k]|^2)
float mel_energies[N_MELS];
float mfcc_matrix[N_MFCC][NUM_FRAMES]; // Shared Shape: (13, 49)

// Rolling variables
float dc_x1 = 0, dc_y1 = 0;
const float DC_R = 0.995f;
int trigger_streak = 0;
volatile int warmup_frames = 0;
const int WARMUP_REQUIRED = 50; // ~1.0 sec of audio before inference starts
volatile uint32_t cooldown_until = 0;

// Dual-core sync variables
portMUX_TYPE matrix_mux = portMUX_INITIALIZER_UNLOCKED;
volatile float latest_rms = 0.0f;
volatile int new_frames_since_inference = 0;

// CPU profiling variables
volatile uint32_t core0_active_us = 0;
volatile uint32_t core1_active_us = 0;
uint32_t last_cpu_report_ms = 0;

// Recording protocol variables
volatile bool recording_mode = false;
volatile bool record_done = false;
int16_t* record_buffer = nullptr;
int record_target_samples = 0;
int record_samples_collected = 0;

// --- ASR STREAMING & PRE-ROLL BUFFER (200ms = 10 hops * 320 samples = 6.4KB) ---
#define PREROLL_HOPS    10
#define PREROLL_SAMPLES (PREROLL_HOPS * HOP_SAMPLES)
int16_t preroll_buffer[PREROLL_SAMPLES];
volatile int preroll_head = 0;
volatile bool is_streaming = false;
QueueHandle_t audio_stream_queue = nullptr;

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
    resolver.AddMean();      // GlobalAvgPool2D
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
    Serial.printf("Input dims: %d [%d,%d,%d,%d] scale:%.5f zero:%d\n",
                  input_tensor->dims->size,
                  input_tensor->dims->data[0], input_tensor->dims->data[1],
                  input_tensor->dims->data[2], input_tensor->dims->data[3],
                  input_tensor->params.scale, input_tensor->params.zero_point);
    Serial.printf("Output dims: %d [%d,%d] scale:%.5f zero:%d\n",
                  output_tensor->dims->size,
                  output_tensor->dims->data[0], output_tensor->dims->data[1],
                  output_tensor->params.scale, output_tensor->params.zero_point);
}

void setup_i2s() {
    i2s_config_t cfg = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 320,  // Exactly our hop size
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
    for (int i = 0; i < 150; i++) i2s_read(I2S_PORT, trash, sizeof(trash), &bytes_read, portMAX_DELAY);
}

void compute_mfcc(int16_t* new_samples, int hop_size) {
    // 1. Shift audio frame by HOP_SAMPLES (320)
    int hop = 320;
    int win = 512;
    for (int i = 0; i < win - hop; i++) {
        audio_frame[i] = audio_frame[i + hop];
    }
    // 2. Append new samples, scaled to nominal [-1.0f, 1.0f] range matching Python librosa
    for (int i = 0; i < hop; i++) {
        audio_frame[win - hop + i] = (float)new_samples[i] / 32768.0f;
    }

    // 3. FFT Prepare (Apply precomputed periodic Hann window into complex interleaved buffer)
    for (int i = 0; i < win; i++) {
        fft_buffer[2 * i] = audio_frame[i] * hann_window[i];
        fft_buffer[2 * i + 1] = 0.0f;
    }

    // 4. Hardware-accelerated Radix-2 FFT via ESP-DSP (Xtensa assembly)
    dsps_fft2r_fc32(fft_buffer, N_FFT);
    dsps_bit_rev2r_fc32(fft_buffer, N_FFT);

    // 5. Compute Power Spectrum (|X[k]|^2 = Re^2 + Im^2) for positive frequencies (257 bins)
    for (int k = 0; k < (N_FFT / 2 + 1); k++) {
        float r = fft_buffer[2 * k];
        float im = fft_buffer[2 * k + 1];
        power_spectrum[k] = r * r + im * im;
    }

    // 6. Mel Filterbank Integration
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

    // 7. DCT (Matrix Multiply) to get 13 MFCCs, Normalize & Shift
    float current_frame_mfcc[N_MFCC];
    for (int c = 0; c < N_MFCC; c++) {
        float sum = 0.0f;
        for (int m = 0; m < N_MELS; m++) {
            sum += dct_basis[c][m] * mel_energies[m];
        }
        current_frame_mfcc[c] = (sum - mfcc_mean[c]) / mfcc_std[c];
    }

    // 8. Atomic update of rolling matrix
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

// Audio Task pinned to Core 0: continuous I2S capture + MFCC extraction
void audio_task(void* pvParameters) {
    int32_t raw32[640];
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

            // Maintain circular pre-roll buffer (200ms)
            for (int i = 0; i < 320; i++) {
                preroll_buffer[(preroll_head + i) % PREROLL_SAMPLES] = pcm[i];
            }
            preroll_head = (preroll_head + 320) % PREROLL_SAMPLES;

            // Stream live PCM chunk over audio queue if active
            if (is_streaming && audio_stream_queue != nullptr) {
                xQueueSend(audio_stream_queue, pcm, 0);
            }

            if (recording_mode && record_buffer != nullptr) {
                memcpy(&record_buffer[record_samples_collected], pcm, sizeof(pcm));
                record_samples_collected += 320;
                if (record_samples_collected >= record_target_samples) {
                    record_done = true;
                    recording_mode = false;
                }
            } else {
                compute_mfcc(pcm, 320);
            }
            core0_active_us += (micros() - t_c0);
        } else {
            vTaskDelay(pdMS_TO_TICKS(5));
        }
    }
}

void check_serial_commands() {
    if (Serial.available() > 0) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        if (cmd.startsWith("REC:")) {
            // Do not free a buffer still owned by the audio task or awaiting dump.
            if (recording_mode || record_done) {
                Serial.println("ERR:BUSY");
                return;
            }
            int duration_ms = cmd.substring(4).toInt();
            if (duration_ms <= 0) duration_ms = 1500;
            if (duration_ms > 4000) duration_ms = 4000;

            record_target_samples = (duration_ms * SAMPLE_RATE) / 1000;
            record_target_samples = ((record_target_samples + 319) / 320) * 320;

            if (record_buffer != nullptr) {
                free(record_buffer);
                record_buffer = nullptr;
            }
            record_buffer = (int16_t*)malloc(record_target_samples * sizeof(int16_t));
            if (record_buffer == nullptr) {
                Serial.println("ERR:OUT_OF_MEMORY");
                return;
            }
            record_samples_collected = 0;
            record_done = false;
            recording_mode = true;
        }
    }
}

void stream_to_serial_asr() {
    uint32_t t_start = millis();
    Serial.println("\n===WAKE:NEXUS:STREAM_START===");

    // 1. Send 200ms pre-roll in chronological order
    int head = preroll_head;
    int16_t temp[320];
    for (int h = 0; h < PREROLL_HOPS; h++) {
        int start_idx = (head + h * 320) % PREROLL_SAMPLES;
        for (int i = 0; i < 320; i++) {
            temp[i] = preroll_buffer[(start_idx + i) % PREROLL_SAMPLES];
        }
        Serial.write((const uint8_t*)temp, sizeof(temp));
    }

    if (audio_stream_queue != nullptr) {
        xQueueReset(audio_stream_queue);
    }
    is_streaming = true;

    // 2. Stream live audio from queue until silence for >1.2s or max 5.0s
    int silence_hops = 0;
    int16_t chunk[320];

    while (millis() - t_start < 5000) {
        if (xQueueReceive(audio_stream_queue, chunk, pdMS_TO_TICKS(50)) == pdTRUE) {
            Serial.write((const uint8_t*)chunk, sizeof(chunk));

            if (latest_rms < VAD_THRESHOLD) {
                silence_hops++;
            } else {
                silence_hops = 0;
            }

            if (silence_hops >= 50) { // 1.0s silence
                break;
            }
        }
    }

    is_streaming = false;
    Serial.flush();
    Serial.println("\n===STREAM_END===");
    cooldown_until = millis() + 1000;
}

void setup() {
    WiFi.mode(WIFI_OFF);
    Serial.setTxBufferSize(4096);
    Serial.setRxBufferSize(1024);
    Serial.begin(921600);
    delay(500);
    Serial.println("Initialzing Dual-Core Real-Time KWS System...");
    setup_tflite();
    setup_i2s();

    audio_stream_queue = xQueueCreate(20, 320 * sizeof(int16_t));

    // Initialize ESP-DSP FFT tables & precompute periodic Hann window
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
        "AudioTask",
        8192,
        NULL,
        2,        // Priority 2
        NULL,
        0         // Core 0 (PRO_CPU)
    );

    Serial.println("System Ready. Listening for Wake Word!");
    Serial.println("Commands: REC:<milliseconds>  -- record raw PCM to serial");
}

void loop() {
    check_serial_commands();

    // Handle dump from recording command
    if (record_done) {
        record_done = false;
        if (record_buffer != nullptr) {
            Serial.printf("===AUDIO_START:%d===\n", record_samples_collected);
            Serial.flush();
            Serial.write((const uint8_t*)record_buffer, record_samples_collected * sizeof(int16_t));
            Serial.flush();
            Serial.println("===AUDIO_END===");
            Serial.flush();

            free(record_buffer);
            record_buffer = nullptr;
        }
        return;
    }

    if (recording_mode) {
        vTaskDelay(pdMS_TO_TICKS(10));
        return;
    }

    // Periodic CPU telemetry (reported every second regardless of idle/active state)
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
        vTaskDelay(pdMS_TO_TICKS(10));
        return;
    }

    // Warmup check
    if (warmup_frames < WARMUP_REQUIRED) {
        new_frames_since_inference = 0;
        vTaskDelay(pdMS_TO_TICKS(10));
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

    // Only run inference if at least 2 new audio hops (40ms) entered the rolling matrix
    if (new_frames_since_inference < 2) {
        vTaskDelay(pdMS_TO_TICKS(5));
        return;
    }

    uint32_t t_c1 = micros();

    // 1. Copy atomic snapshot of MFCC matrix
    float local_matrix[N_MFCC][NUM_FRAMES];
    portENTER_CRITICAL(&matrix_mux);
    memcpy(local_matrix, mfcc_matrix, sizeof(mfcc_matrix));
    new_frames_since_inference = 0;
    portEXIT_CRITICAL(&matrix_mux);

    // 2. Quantize into INT8 Tensor Arena
    float t_scale = input_tensor->params.scale;
    int32_t t_zero = input_tensor->params.zero_point;

    for (int m = 0; m < N_MFCC; m++) {
        for (int f = 0; f < NUM_FRAMES; f++) {
            float val = local_matrix[m][f];
            int8_t quantized = (int8_t)max(min((int32_t)round(val / t_scale) + t_zero, 127), -128);
            input_tensor->data.int8[m * NUM_FRAMES + f] = quantized;
        }
    }

    // 3. Run Model Inference on Core 1
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Invoke failed!");
        return;
    }
    core1_active_us += (micros() - t_c1);

    // 4. Check Wake Word Confidence
    int8_t wakeword_score = output_tensor->data.int8[1];
    float o_scale = output_tensor->params.scale;
    int32_t o_zero = output_tensor->params.zero_point;
    float prob = ((int)wakeword_score - (int)o_zero) * o_scale;
    if (prob < 0.0f) prob = 0.0f;
    if (prob > 1.0f) prob = 1.0f;

    Serial.printf("Prob: %.2f | Streak: %d | RMS: %.0f\n", prob, trigger_streak, rms);

    // 5. Detection Trigger:
    bool triggered = false;
    if (prob >= 0.85f) {
        triggered = true;
    } else if (prob >= 0.70f) {
        trigger_streak++;
        if (trigger_streak >= 3) {
            triggered = true;
        }
    } else {
        trigger_streak = 0;
    }

    if (triggered) {
        Serial.printf("\n>>>> WAKE WORD DETECTED! <<<< (Confidence: %.2f | RMS: %.0f)\n", prob, rms);
        trigger_streak = 0;
        stream_to_serial_asr();
    }
}
