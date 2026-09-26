#!/usr/bin/env python3
"""
download_speech_commands.py — Download Google Speech Commands v2 dataset
for use as real human negative samples in KWS training.

Downloads ~2.3GB, extracts to dataset/speech_commands_v2/
Only needs to run once.
"""

import os
import sys
import tarfile
import urllib.request
import hashlib

DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
DATASET_DIR = "dataset/speech_commands_v2"
ARCHIVE_PATH = "dataset/speech_commands_v0.02.tar.gz"
EXPECTED_MD5 = "6b74f3901214cb2c2934e98196829835"


def download_with_progress(url, dest):
    """Download file with progress bar."""
    print(f"Downloading: {url}")
    print(f"Destination: {dest}")

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        block_size = 1024 * 1024  # 1MB blocks

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            while True:
                block = response.read(block_size)
                if not block:
                    break
                f.write(block)
                downloaded += len(block)
                if total > 0:
                    pct = downloaded / total * 100
                    mb_done = downloaded / (1024 * 1024)
                    mb_total = total / (1024 * 1024)
                    bar_len = 40
                    filled = int(bar_len * downloaded / total)
                    bar = "=" * filled + "-" * (bar_len - filled)
                    print(f"\r  [{bar}] {pct:5.1f}% ({mb_done:.0f}/{mb_total:.0f} MB)", end="", flush=True)
                else:
                    print(f"\r  Downloaded {downloaded / (1024*1024):.0f} MB", end="", flush=True)

    print()
    return dest


def verify_md5(filepath, expected):
    """Verify file MD5 checksum."""
    print(f"Verifying MD5 checksum...")
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)
    actual = md5.hexdigest()
    if actual == expected:
        print(f"  Checksum OK: {actual}")
        return True
    else:
        print(f"  WARNING: Expected {expected}, got {actual}")
        return False


def extract_archive(archive, dest):
    """Extract tar.gz archive."""
    print(f"Extracting to {dest}/ ...")
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        total = len(members)
        for i, member in enumerate(members):
            tar.extract(member, dest)
            if i % 5000 == 0:
                print(f"  Extracted {i}/{total} files...")
    print(f"  Done! {total} files extracted.")


def print_summary(dest):
    """Print dataset summary."""
    print(f"\nDataset summary ({dest}):")
    categories = sorted([d for d in os.listdir(dest) if os.path.isdir(os.path.join(dest, d))])
    total_files = 0
    for cat in categories:
        cat_dir = os.path.join(dest, cat)
        n = len([f for f in os.listdir(cat_dir) if f.endswith(".wav")])
        total_files += n
        print(f"  {cat:25s}: {n:6d} files")
    print(f"  {'TOTAL':25s}: {total_files:6d} files")
    return categories


def main():
    # Check if already downloaded
    if os.path.isdir(DATASET_DIR):
        categories = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
        if len(categories) > 20:
            print(f"Dataset already exists at {DATASET_DIR}/ ({len(categories)} categories)")
            print_summary(DATASET_DIR)
            print("\nTo re-download, delete the directory first.")
            return

    # Download
    if os.path.exists(ARCHIVE_PATH):
        print(f"Archive already downloaded: {ARCHIVE_PATH}")
    else:
        download_with_progress(DATASET_URL, ARCHIVE_PATH)

    # Verify
    verify_md5(ARCHIVE_PATH, EXPECTED_MD5)

    # Extract
    extract_archive(ARCHIVE_PATH, DATASET_DIR)

    # Summary
    print_summary(DATASET_DIR)

    # Cleanup archive
    if os.path.exists(ARCHIVE_PATH):
        os.remove(ARCHIVE_PATH)
        print("Archive removed after extraction to preserve disk space.")

    print("\nReady for training! Run: python3 train_kws_v4.py")


if __name__ == "__main__":
    main()
