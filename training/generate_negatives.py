from pathlib import Path
import asyncio
import os
import edge_tts

WORDS = [
    # Phonetic hard negatives (rhymes, prefix matches, suffix matches)
    "Texas", "Lexus", "Alexis", "Plexus", "Census", "Next", "Necklace",
    "Netflix", "Nectar", "Exit", "Access", "Excess", "Fix us", "Mix us",
    "Tax us", "Flexes", "Boxes", "Foxes", "Sixes", "Neck", "Exes", "Hexes",
    "Texas road", "Next one", "Access denied", "Network", "Never", "Nest",
    # Common household / assistant words
    "Hello", "Computer", "Alexa", "Siri", "Google", "Hey Google", "Hey Siri",
    "Weather", "Music", "Play", "Pause", "Stop", "Volume", "Turn on", "Turn off",
    "Lights", "Fan", "Switch", "Living room", "Bedroom", "Kitchen",
    # Numbers & commands
    "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
    "Yes", "No", "Okay", "Cancel", "Start", "Reset", "Repeat", "Confirm",
    # Conversational babble / speech
    "What time is it", "How are you doing", "Can you hear me", "Good morning",
    "Good night", "I am talking right now", "Testing the microphone",
    "This is random speaking", "Blabbering words", "Just talking casually",
    "Open the door", "Close the window", "Send a message", "Call my phone"
]

VOICES = [
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-GB-RyanNeural",
    "en-GB-SoniaNeural",
    "en-AU-NatashaNeural",
    "en-CA-ClaraNeural",
    "en-IN-PrabhatNeural",
    "en-IN-NeerjaNeural"
]

OUT_DIR = str(Path(__file__).resolve().parents[1] / "dataset/negatives_expanded")

async def generate(word, voice, path):
    try:
        c = edge_tts.Communicate(word, voice)
        await c.save(path)
    except Exception as e:
        print(f"Error {word} / {voice}: {e}")

async def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = []
    idx = 0
    for word in WORDS:
        # Pick 2-3 voices per word
        for v_i in range(len(VOICES)):
            if (hash(word) + v_i) % 3 == 0: # deterministic selection
                voice = VOICES[v_i]
                safe_word = "".join(c for c in word if c.isalnum() or c == '_').lower()
                filepath = os.path.join(OUT_DIR, f"{safe_word}_{voice}.mp3")
                if not os.path.exists(filepath):
                    tasks.append(generate(word, voice, filepath))
                    idx += 1
    print(f"Generating {len(tasks)} negative speech files...")
    # Run in batches of 10 concurrent requests
    for i in range(0, len(tasks), 10):
        batch = tasks[i:i+10]
        await asyncio.gather(*batch)
        print(f"  Generated {min(i+10, len(tasks))}/{len(tasks)}...")
    print("Done generating negatives!")

if __name__ == '__main__':
    asyncio.run(main())
