import asyncio
import os
import edge_tts

# Wake Word & Hard Negatives definition
WAKE_WORD = "Nexus"
HARD_NEGATIVES = ["Texas", "Next", "Neck", "Hexes", "Flexes", "Excess", "Alexis", "Exes"]
UNKNOWN_WORDS = ["Hello", "Computer", "Weather", "Music", "Play", "Pause", "Stop", "Turn on the lights", "What time is it"]

# Output structures
DATASET_DIR = "dataset"
CATEGORIES = {
    "positive": ["en-US-ChristopherNeural", "en-GB-RyanNeural", "en-CA-LiamNeural", "en-AU-WilliamNeural", "en-US-AvaNeural", "en-GB-SoniaNeural"],
    "negative": ["en-US-GuyNeural", "en-CA-ClaraNeural", "en-IE-ConnorNeural"],
    "unknown": ["en-US-AnaNeural", "en-US-EricNeural"]
}

async def generate_audio(text, voice, rate_shift, pitch_shift, filepath):
    rate_str = f"{rate_shift:+}%"
    pitch_str = f"{pitch_shift:+}Hz"
    
    communicate = edge_tts.Communicate(text, voice, rate=rate_str, pitch=pitch_str)
    await communicate.save(filepath)

async def main():
    os.makedirs(f"{DATASET_DIR}/positive", exist_ok=True)
    os.makedirs(f"{DATASET_DIR}/negative", exist_ok=True)
    os.makedirs(f"{DATASET_DIR}/unknown", exist_ok=True)
    
    print(f"Generating dataset for Wake Word: {WAKE_WORD}")
    
    # 1. Generate Positives (Nexus)
    print("Generating POSITIVE samples...")
    count = 0
    for voice in CATEGORIES["positive"]:
        for rate in [-15, 0, 15]:
            for pitch in [-10, 0, 10]:
                filename = f"{DATASET_DIR}/positive/nexus_{voice}_{rate}_{pitch}.wav"
                await generate_audio(WAKE_WORD, voice, rate, pitch, filename)
                count += 1
    print(f" -> Generated {count} positive samples.")

    # 2. Generate Hard Negatives
    print("Generating HARD NEGATIVE samples...")
    count = 0
    for neg_word in HARD_NEGATIVES:
        for voice in CATEGORIES["negative"]:
             for rate in [-10, 0, 10]: # slightly fewer variations to save time
                filename = f"{DATASET_DIR}/negative/{neg_word}_{voice}_{rate}.wav"
                await generate_audio(neg_word, voice, rate, 0, filename)
                count += 1
    print(f" -> Generated {count} hard negative samples.")

    # 3. Generate Unknown Words
    print("Generating UNKNOWN samples...")
    count = 0
    for unk_word in UNKNOWN_WORDS:
        for voice in CATEGORIES["unknown"]:
            safe_word = unk_word.replace(" ", "_")
            filename = f"{DATASET_DIR}/unknown/{safe_word}_{voice}.wav"
            await generate_audio(unk_word, voice, 0, 0, filename)
            count += 1
    print(f" -> Generated {count} unknown samples.")
    
    print("\nDataset generation complete! (Ready for Phase 3 Data Augmentation)")

if __name__ == "__main__":
    asyncio.run(main())