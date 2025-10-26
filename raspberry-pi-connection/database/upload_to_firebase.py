#!/usr/bin/env python3
"""
Standalone Firebase Storage Uploader

Run this AFTER deployment completes to upload files to Firebase Storage.
This way Firebase never interferes with GPIO.

Usage:
  python3 upload_to_firebase.py /path/to/run_folder

Example:
  python3 upload_to_firebase.py /home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/results/run_20251026_121519

Or upload all recent runs:
  python3 upload_to_firebase.py --all
"""

import sys
import os
import glob
from firebase_uploader import FirebaseUploader


def upload_single_run(run_folder):
    """Upload a single run folder"""
    if not os.path.exists(run_folder):
        print(f"[ERROR] Run folder not found: {run_folder}")
        return False

    print(f"\n{'='*70}")
    print(f"UPLOADING TO FIREBASE STORAGE")
    print(f"{'='*70}\n")

    uploader = FirebaseUploader()

    if not uploader.initialized:
        print("[ERROR] Failed to initialize Firebase. Check service account key.")
        return False

    uploaded = uploader.upload_run_folder(run_folder)

    if uploaded:
        print(f"\n[SUCCESS] Uploaded {len(uploaded)} files from {os.path.basename(run_folder)}")
        print("\nView the files at:")
        print("https://console.firebase.google.com/project/traffic-ppo-pi/storage")
        return True
    else:
        print("[ERROR] Upload failed")
        return False


def upload_all_recent(results_dir, limit=5):
    """Upload the most recent runs"""
    run_folders = glob.glob(os.path.join(results_dir, "run_*"))
    run_folders.sort(reverse=True)  # Most recent first

    if not run_folders:
        print(f"[ERROR] No run folders found in {results_dir}")
        return

    print(f"\nFound {len(run_folders)} run folders. Uploading {min(limit, len(run_folders))} most recent...\n")

    uploader = FirebaseUploader()

    if not uploader.initialized:
        print("[ERROR] Failed to initialize Firebase. Check service account key.")
        return

    success_count = 0
    for i, run_folder in enumerate(run_folders[:limit], 1):
        print(f"\n[{i}/{min(limit, len(run_folders))}] Uploading {os.path.basename(run_folder)}...")
        uploaded = uploader.upload_run_folder(run_folder)
        if uploaded:
            success_count += 1

    print(f"\n{'='*70}")
    print(f"[COMPLETE] Uploaded {success_count}/{min(limit, len(run_folders))} runs")
    print(f"{'='*70}\n")
    print("View the files at:")
    print("https://console.firebase.google.com/project/traffic-ppo-pi-19ab7/storage")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 upload_to_firebase.py /path/to/run_folder")
        print("  python3 upload_to_firebase.py --all")
        print("\nExample:")
        print("  python3 upload_to_firebase.py /home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/results/run_20251026_121519")  
        sys.exit(1)

    if sys.argv[1] == '--all':
        results_dir = '/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/results'
        limit = 5
        if len(sys.argv) > 2:
            try:
                limit = int(sys.argv[2])
            except ValueError:
                pass
        upload_all_recent(results_dir, limit)
    else:
        run_folder = sys.argv[1]
        upload_single_run(run_folder)


if __name__ == "__main__":
    main()