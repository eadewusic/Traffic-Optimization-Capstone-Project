#!/usr/bin/env python3
"""
Standalone Firebase Storage Uploader

Run this AFTER deployment completes to upload files to Firebase Storage.
This way Firebase never interferes with GPIO.

Uploads both run folders and comparison analysis files automatically.

Usage:
  python3 upload_to_firebase.py                    # Upload all new runs + comparisons
  python3 upload_to_firebase.py --all              # Upload all (including already uploaded)
  python3 upload_to_firebase.py /path/to/run       # Upload specific run
"""

import sys
import os
import glob
import json
from firebase_uploader import FirebaseUploader


UPLOAD_TRACKER_FILE = '/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/results/.uploaded_runs.json'


def load_uploaded_runs():
    """Load list of already uploaded runs"""
    if os.path.exists(UPLOAD_TRACKER_FILE):
        try:
            with open(UPLOAD_TRACKER_FILE, 'r') as f:
                return set(json.load(f))
        except:
            return set()
    return set()


def save_uploaded_run(run_name):
    """Mark a run as uploaded"""
    uploaded = load_uploaded_runs()
    uploaded.add(run_name)
    try:
        with open(UPLOAD_TRACKER_FILE, 'w') as f:
            json.dump(list(uploaded), f, indent=2)
    except Exception as e:
        print(f"[WARNING] Could not save upload tracker: {e}")


def upload_single_run(run_folder, force=False):
    """Upload a single run folder"""
    if not os.path.exists(run_folder):
        print(f"[ERROR] Run folder not found: {run_folder}")
        return False
    
    if not os.path.isdir(run_folder):
        print(f"[SKIP] Not a directory: {run_folder}")
        return False
    
    run_name = os.path.basename(run_folder)
    
    # Skip if already uploaded (unless force=True)
    if not force:
        uploaded = load_uploaded_runs()
        if run_name in uploaded:
            print(f"[SKIP] Already uploaded: {run_name}")
            return False
    
    print(f"\n{'='*70}")
    print(f"UPLOADING: {run_name}")
    print(f"{'='*70}\n")
    
    uploader = FirebaseUploader()
    
    if not uploader.initialized:
        print("[ERROR] Failed to initialize Firebase. Check service account key.")
        return False
    
    uploaded_files = uploader.upload_run_folder(run_folder)
    
    if uploaded_files:
        print(f"\n[SUCCESS] Uploaded {len(uploaded_files)} files")
        save_uploaded_run(run_name)
        return True
    else:
        print("[ERROR] Upload failed")
        return False


def upload_all_new_runs(results_dir):
    """Upload all runs that haven't been uploaded yet"""
    run_folders = glob.glob(os.path.join(results_dir, "run_*"))
    run_folders.sort(reverse=True)  # Most recent first
    
    # Also find comparison files
    comparison_files = glob.glob(os.path.join(results_dir, "comparison_analysis_*.txt"))
    comparison_files.sort(reverse=True)
    
    if not run_folders and not comparison_files:
        print(f"[ERROR] No run folders or comparison files found in {results_dir}")
        return
    
    uploaded_runs = load_uploaded_runs()
    new_runs = [rf for rf in run_folders if os.path.basename(rf) not in uploaded_runs]
    new_comparisons = [cf for cf in comparison_files if os.path.basename(cf) not in uploaded_runs]
    
    if not new_runs and not new_comparisons:
        print("\n[INFO] All runs and comparisons already uploaded to Firebase!")
        print(f"[INFO] Total runs: {len(run_folders)}")
        print(f"[INFO] Total comparisons: {len(comparison_files)}")
        print("\nUse 'python3 upload_to_firebase.py --all' to re-upload everything")
        return
    
    print(f"\n{'='*70}")
    print(f"UPLOADING NEW RUNS TO FIREBASE")
    print(f"{'='*70}")
    print(f"Total runs: {len(run_folders)}")
    print(f"Total comparisons: {len(comparison_files)}")
    print(f"Already uploaded: {len(uploaded_runs)}")
    print(f"New runs to upload: {len(new_runs)}")
    print(f"New comparisons to upload: {len(new_comparisons)}\n")
    
    uploader = FirebaseUploader()
    
    if not uploader.initialized:
        print("[ERROR] Failed to initialize Firebase. Check service account key.")
        return
    
    success_count = 0
    
    # Upload run folders
    for i, run_folder in enumerate(new_runs, 1):
        run_name = os.path.basename(run_folder)
        print(f"[{i}/{len(new_runs)}] Uploading run: {run_name}...")
        uploaded_files = uploader.upload_run_folder(run_folder)
        if uploaded_files:
            save_uploaded_run(run_name)
            success_count += 1
            print(f"  Uploaded {len(uploaded_files)} files")
        else:
            print(f"  Failed")
    
    # Upload comparison files
    for i, comparison_file in enumerate(new_comparisons, 1):
        comp_name = os.path.basename(comparison_file)
        print(f"[{i}/{len(new_comparisons)}] Uploading comparison: {comp_name}...")
        url = uploader.upload_comparison(comparison_file)
        if url:
            save_uploaded_run(comp_name)
            success_count += 1
            print(f"  Uploaded")
        else:
            print(f"  Failed")
    
    print(f"\n{'='*70}")
    print(f"[COMPLETE] Uploaded {success_count}/{len(new_runs) + len(new_comparisons)} items")
    print(f"{'='*70}\n")
    print("View files at:")
    print("https://console.firebase.google.com/project/traffic-ppo-pi-19ab7/storage")


def upload_all_runs_force(results_dir):
    """Upload ALL runs and comparisons, even if already uploaded (creates duplicates in Firebase)"""
    run_folders = glob.glob(os.path.join(results_dir, "run_*"))
    run_folders.sort(reverse=True)
    
    comparison_files = glob.glob(os.path.join(results_dir, "comparison_analysis_*.txt"))
    comparison_files.sort(reverse=True)
    
    if not run_folders and not comparison_files:
        print(f"[ERROR] No run folders or comparison files found in {results_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"UPLOADING ALL RUNS AND COMPARISONS (INCLUDING DUPLICATES)")
    print(f"{'='*70}")
    print(f"Total runs: {len(run_folders)}")
    print(f"Total comparisons: {len(comparison_files)}\n")
    
    uploader = FirebaseUploader()
    
    if not uploader.initialized:
        print("[ERROR] Failed to initialize Firebase. Check service account key.")
        return
    
    success_count = 0
    
    # Upload run folders
    for i, run_folder in enumerate(run_folders, 1):
        run_name = os.path.basename(run_folder)
        print(f"[{i}/{len(run_folders)}] Uploading run: {run_name}...")
        uploaded_files = uploader.upload_run_folder(run_folder)
        if uploaded_files:
            save_uploaded_run(run_name)
            success_count += 1
            print(f"  Uploaded {len(uploaded_files)} files")
        else:
            print(f"  Failed")
    
    # Upload comparison files
    for i, comparison_file in enumerate(comparison_files, 1):
        comp_name = os.path.basename(comparison_file)
        print(f"[{i}/{len(comparison_files)}] Uploading comparison: {comp_name}...")
        url = uploader.upload_comparison(comparison_file)
        if url:
            save_uploaded_run(comp_name)
            success_count += 1
            print(f"  Uploaded")
        else:
            print(f"  Failed")
    
    print(f"\n{'='*70}")
    print(f"[COMPLETE] Uploaded {success_count}/{len(run_folders) + len(comparison_files)} items")
    print(f"{'='*70}\n")


def main():
    results_dir = '/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/results'
    
    # No arguments - upload all new runs
    if len(sys.argv) == 1:
        upload_all_new_runs(results_dir)
    
    # --all flag - upload everything including duplicates
    elif sys.argv[1] == '--all':
        upload_all_runs_force(results_dir)
    
    # Specific run folder
    else:
        run_folder = sys.argv[1]
        upload_single_run(run_folder, force=True)


if __name__ == "__main__":
    main()
