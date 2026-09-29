# Research Foundations and References

**Project**: Low Latency and Efficient Voice Activator for Edge Devices (Nexus KWS)  
**Hackathon / Track**: Smart India Hackathon (SIH) — Hardware / Embedded AI Edition  

This document details the academic literature, engineering standards, open-source codebases, and hardware specifications underpinning the architectural design and implementation of Nexus KWS.

---

## 1. Academic Research Papers

### 1.1 TinyML Keyword Spotting & Model Architecture
1. **Zhang, Y., Suda, N., Lai, L., & Chandra, V. (2017).**  
   *"Hello Edge: Keyword Spotting on Microcontrollers."*  
   *arXiv preprint arXiv:1711.07128.*  
   - **Link**: https://arxiv.org/abs/1711.07128  
   - **Relevance**: Proved that Depthwise-Separable Convolutional Neural Networks (DS-CNN) achieve superior memory-accuracy trade-offs on microcontrollers compared to standard DNNs, RNNs, and standard CNNs. Direct foundation for our 1,294-parameter DS-CNN topology.

2. **Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., & Adam, H. (2017).**  
   *"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications."*  
   *arXiv preprint arXiv:1704.04861.*  
   - **Link**: https://arxiv.org/abs/1704.04861  
   - **Relevance**: Established the factorization of standard convolutions into depthwise and pointwise layers, reducing computation by $\frac{1}{N} + \frac{1}{D_k^2}$ (~8x to 9x reduction in MACs).

3. **David, R., Duke, P., Jain, A., Janapa Reddi, V., Jeffries, N., Li, J., Kreeger, N., Njoroge, I., Ting, P., Wang, Z., Wei, M., & Warden, P. (2021).**  
   *"TensorFlow Lite Micro: Embedded Machine Learning on TinyML Systems."*  
   *Proceedings of Machine Learning and Systems (MLSys), 3, 800-811.*  
   - **Link**: https://proceedings.mlsys.org/paper_files/paper/2021/file/8a1e808b55fde833a45a1aacfd570775-Paper.pdf  
   - **Relevance**: Architecture reference for static memory arenas (`TensorArena`), elimination of runtime dynamic memory allocation (`malloc`), and custom operator registration via `MicroMutableOpResolver`.

### 1.2 Integer Quantization & Numerical Optimization
4. **Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H., & Kalenichenko, D. (2018).**  
   *"Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference."*  
   *IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2018), pp. 2704-2713.*  
   - **Link**: https://arxiv.org/abs/1712.05877  
   - **Relevance**: Theoretical basis for our symmetric 8-bit integer quantization scheme (INT8 weights, activations, and accumulator scaling), compressing the DS-CNN binary to 10.2 KB.

### 1.3 Audio DSP, Spectral Analysis & Feature Extraction
5. **Davis, S., & Mermelstein, P. (1980).**  
   *"Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences."*  
   *IEEE Transactions on Acoustics, Speech, and Signal Processing, 28(4), 357-366.*  
   - **Link**: https://ieeexplore.ieee.org/document/1163420  
   - **Relevance**: Foundational study establishing Mel-Frequency Cepstral Coefficients (MFCCs) as the optimal acoustic representation for speech discrimination over linear prediction coefficients.

6. **Slaney, M. (1998).**  
   *"Auditory Toolbox: A MATLAB Toolbox for Auditory Modeling Work."*  
   *Interval Research Corporation Technical Report #1998-010.*  
   - **Link**: https://engineering.purdue.edu/~malcolm/interval/1998-010/  
   - **Relevance**: Formalized the triangular, area-normalized Mel filterbank definition used in both our Python training pipeline (`librosa.filters.mel`) and ESP-DSP C++ implementation.

---

## 2. Public Datasets & Acoustic Benchmarks

7. **Warden, P. (2018).**  
   *"Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition."*  
   *arXiv preprint arXiv:1804.03209.*  
   - **Dataset URL**: http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz  
   - **Relevance**: Primary source for non-wake human speech evaluation (~3,000 clips sampled across 35 vocabulary words), used to verify zero false positives on common English commands.

8. **Panayotov, V., Chen, G., Povey, D., & Khudanpur, S. (2015).**  
   *"Librispeech: An ASR corpus based on public domain audio books."*  
   *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2015), pp. 5206-5210.*  
   - **Dataset URL**: https://www.openslr.org/12/  
   - **Relevance**: Source corpus for baseline acoustic model training across English phoneme distributions.

---

## 3. Open-Source Frameworks & Toolkits

9. **Espressif ESP-DSP Library**  
   *Official Digital Signal Processing Library for Espressif Chips.*  
   - **Repository**: https://github.com/espressif/esp-dsp  
   - **Documentation**: https://docs.espressif.com/projects/esp-dsp/en/latest/esp32/  
   - **Relevance**: Used for hardware-accelerated Radix-2 assembly FFT (`dsps_fft2r_fc32_ae32_`) and bit-reversal tables, cutting FFT execution time from ~8,200 µs to ~45 µs.

10. **TensorFlow Lite for Microcontrollers (TFLM)**  
    *Deep Learning inference engine for embedded microcontrollers.*  
    - **Repository**: https://github.com/tensorflow/tflite-micro  
    - **Arduino ESP32 Port**: https://github.com/tanakamasayuki/Arduino_TensorFlowLite_ESP32  
    - **Relevance**: Runtime engine executing quantized INT8 DS-CNN inference on Xtensa Core 1.

11. **Vosk Offline Speech Recognition API (Alpha Cephei)**  
    *Kaldi-based lightweight streaming speech recognition toolkit.*  
    - **Repository**: https://github.com/alphacep/vosk-api  
    - **Acoustic Model**: `vosk-model-small-en-in-0.4` (Indian English Accent Model, 40 MB)  
    - **Model Hub**: https://alphacephei.com/vosk/models  
    - **Relevance**: Hosts the streaming TCP speech transcription daemon for Phase 6 ASR handoff.

12. **FreeRTOS Kernel**  
    *Real-Time Operating System for Microcontrollers.*  
    - **Documentation**: https://www.freertos.org/Documentation/RTOS_book.html  
    - **Relevance**: Used for dual-core task affinity (`xTaskCreatePinnedToCore`), inter-core non-blocking queues (`audio_stream_queue`), and critical sections protecting shared feature matrices.

---

## 4. Hardware Datasheets & Protocols

13. **TDK InvenSense INMP441 MEMS Microphone Datasheet**  
    *Omnidirectional Microphone with I2S Digital Output (Rev 1.1).*  
    - **Datasheet URL**: https://invensense.tdk.com/wp-content/uploads/2015/02/INMP441.pdf  
    - **Specifications Verified**:
      - SNR: 61 dBA
      - Sensitivity: -26 dBFS
      - Frequency Response: 60 Hz to 15 kHz
      - Current Consumption: 1.4 mA active
      - Output: 24-bit I2S data in 32-bit slot

14. **Espressif ESP32 Technical Reference Manual**  
    *ESP32 Series 32-bit Microprocessors (Xtensa LX6).*  
    - **Manual URL**: https://www.espressif.com/sites/default/files/documentation/esp32_technical_reference_manual_en.pdf  
    - **Key Chapters Consulted**:
      - Chapter 12: *I2S Controller (I2S DMA Ring Buffers & Timing)*
      - Chapter 3: *DMA Controller*
      - Chapter 2: *Wi-Fi Architecture, Low-Power Modes & Power Backoff*

15. **Philips Semiconductors I2S Bus Specification**  
    *The I2S Bus Specification (February 1986, revised June 1996).*  
    - **Standard URL**: https://www.nxp.com/docs/en/user-guide/UM11732.pdf  
    - **Relevance**: Governs the 3-wire physical bus timing (SCK, WS, SD) and MSB-first left-justified serial data frame structure.
