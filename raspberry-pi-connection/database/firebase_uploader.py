"""
Firebase Cloud Storage Integration for Traffic Control System

Handles uploading deployment logs, visualizations, and reports to Firebase Storage.
Requires firebase-admin SDK and service account credentials.

Setup:
1. Place Firebase service account JSON file in the project root
2. Bucket name is taken from Firebase console
"""

import os
from datetime import datetime
from pathlib import Path


class FirebaseUploader:
    """Manages Firebase Storage uploads for traffic control deployment data"""

    def __init__(self, service_account_path=None, storage_bucket=None):
        """
        Initialize Firebase connection

        Args:
            service_account_path: Path to Firebase service account JSON
            storage_bucket: Firebase storage bucket name
        """
        # Import firebase here to avoid threading conflicts with GPIO
        import firebase_admin
        from firebase_admin import credentials, storage

        if service_account_path is None:
            service_account_path = '/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/traffic-ppo-pi-firebase-key.json'

        if storage_bucket is None:
            storage_bucket = 'traffic-ppo-pi-19ab7.firebasestorage.app'

        self.service_account_path = service_account_path
        self.storage_bucket = storage_bucket
        self.initialized = False

        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(service_account_path)
                firebase_admin.initialize_app(cred, {
                    'storageBucket': storage_bucket
                })

            self.bucket = storage.bucket()
            self.initialized = True
            print(f"[FIREBASE] Connected to Storage: {storage_bucket}")

        except Exception as e:
            print(f"[FIREBASE ERROR] Failed to initialize: {e}")
            self.initialized = False

    def upload_file(self, local_path, remote_path=None):
        """
        Upload a single file to Firebase Storage

        Args:
            local_path: Path to local file
            remote_path: Destination path in Firebase (auto-generated if None)

        Returns:
            Public URL of uploaded file, or None if failed
        """
        if not self.initialized:
            return None

        try:
            if not os.path.exists(local_path):
                print(f"[FIREBASE] File not found: {local_path}")
                return None

            if remote_path is None:
                remote_path = os.path.basename(local_path)

            blob = self.bucket.blob(remote_path)
            blob.upload_from_filename(local_path)

            blob.make_public()
            url = blob.public_url

            file_size = os.path.getsize(local_path) / 1024
            print(f"[FIREBASE] Uploaded: {os.path.basename(local_path)} ({file_size:.1f} KB)")

            return url

        except Exception as e:
            print(f"[FIREBASE ERROR] Upload failed for {local_path}: {e}")
            return None

    def upload_run_folder(self, run_folder):
        """
        Upload all files from a deployment run folder

        Args:
            run_folder: Path to run folder (e.g., results/run_20241026_153045)

        Returns:
            Dictionary mapping filenames to public URLs
        """
        if not self.initialized:
            return {}

        run_name = os.path.basename(run_folder)
        uploaded_files = {}

        print(f"\n[FIREBASE] Uploading run: {run_name}")

        # Walk through all files in the run directory and subdirectories
        for root, dirs, files in os.walk(run_folder):
            for filename in files:
                local_path = os.path.join(root, filename)
                
                # Create relative path for Firebase storage
                relative_path = os.path.relpath(local_path, run_folder)
                remote_path = f"deployments/{run_name}/{relative_path}"
                
                url = self.upload_file(local_path, remote_path)

                if url:
                    uploaded_files[relative_path] = url

        if uploaded_files:
            print(f"[FIREBASE] Successfully uploaded {len(uploaded_files)} files")

        return uploaded_files

    def upload_comparison(self, comparison_file):
        """
        Upload comparison analysis file

        Args:
            comparison_file: Path to comparison analysis text file

        Returns:
            Public URL of uploaded file, or None if failed
        """
        if not self.initialized:
            return None

        filename = os.path.basename(comparison_file)
        remote_path = f"comparisons/{filename}"

        return self.upload_file(comparison_file, remote_path)

    def upload_multiple_runs(self, run_folders):
        """
        Upload multiple run folders (used for comparison mode)

        Args:
            run_folders: List of run folder paths

        Returns:
            Dictionary mapping run names to their uploaded files
        """
        if not self.initialized:
            return {}

        all_uploads = {}

        for run_folder in run_folders:
            run_name = os.path.basename(run_folder)
            uploaded = self.upload_run_folder(run_folder)
            all_uploads[run_name] = uploaded

        return all_uploads
