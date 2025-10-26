"""
Firebase Cloud Firestore Integration for Traffic Control System

Handles uploading deployment logs and statistics to Firestore database.
Requires firebase-admin SDK and service account credentials.

Setup:
1. Place Firebase service account JSON file in the project root
2. Enable Firestore in Firebase Console (not Storage!)
3. Update SERVICE_ACCOUNT_PATH if using a different location
"""

import os
import json
from datetime import datetime


class FirebaseUploader:
    """Manages Firestore uploads for traffic control deployment data"""
    
    def __init__(self, service_account_path=None):
        """
        Initialize Firestore connection
        
        Args:
            service_account_path: Path to Firebase service account JSON
        """
        # Import firebase here to avoid threading conflicts with GPIO
        import firebase_admin
        from firebase_admin import credentials, firestore
        
        if service_account_path is None:
            service_account_path = '/home/tpi4/Desktop/Traffic-Optimization-Capstone-Project/traffic-ppo-pi-firebase-key.json'
        
        self.service_account_path = service_account_path
        self.initialized = False
        
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(service_account_path)
                firebase_admin.initialize_app(cred)
            
            self.db = firestore.client()
            self.initialized = True
            print(f"[FIREBASE] Connected to Firestore")
            
        except Exception as e:
            print(f"[FIREBASE ERROR] Failed to initialize: {e}")
            self.initialized = False
    
    def upload_deployment_stats(self, stats, run_id=None):
        """
        Upload deployment statistics to Firestore
        
        Args:
            stats: Statistics dictionary from deployment
            run_id: Optional run identifier (timestamp-based if None)
        
        Returns:
            Document ID if successful, None otherwise
        """
        if not self.initialized:
            return None
        
        try:
            if run_id is None:
                run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Store in deployments collection
            doc_ref = self.db.collection('deployments').document(run_id)
            doc_ref.set(stats)
            
            print(f"[FIREBASE] Uploaded stats for run: {run_id}")
            return run_id
            
        except Exception as e:
            print(f"[FIREBASE ERROR] Upload failed: {e}")
            return None
    
    def upload_comparison_results(self, fixed_stats, ppo_stats, comparison_id=None):
        """
        Upload comparison results to Firestore
        
        Args:
            fixed_stats: Statistics from fixed-timing controller
            ppo_stats: Statistics from PPO controller
            comparison_id: Optional comparison identifier
        
        Returns:
            Document ID if successful, None otherwise
        """
        if not self.initialized:
            return None
        
        try:
            if comparison_id is None:
                comparison_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            comparison_data = {
                'timestamp': datetime.now().isoformat(),
                'fixed_timing': fixed_stats,
                'ppo_powered': ppo_stats,
                'improvement': {
                    'vehicles_cleared': ppo_stats['vehicles_cleared'] - fixed_stats['vehicles_cleared'],
                    'phase_changes': ppo_stats['phase_changes'] - fixed_stats['phase_changes']
                }
            }
            
            # Store in comparisons collection
            doc_ref = self.db.collection('comparisons').document(comparison_id)
            doc_ref.set(comparison_data)
            
            print(f"[FIREBASE] Uploaded comparison: {comparison_id}")
            return comparison_id
            
        except Exception as e:
            print(f"[FIREBASE ERROR] Upload failed: {e}")
            return None
    
    def upload_run_folder(self, run_folder):
        """
        Upload data from a run folder by reading the JSON stats file
        
        Args:
            run_folder: Path to run folder
        
        Returns:
            Document ID if successful, None otherwise
        """
        if not self.initialized:
            return None
        
        try:
            # Read the stats JSON file
            json_path = os.path.join(run_folder, "deployment_stats.json")
            
            if not os.path.exists(json_path):
                print(f"[FIREBASE] No stats file found in {run_folder}")
                return None
            
            with open(json_path, 'r') as f:
                stats = json.load(f)
            
            # Get run ID from folder name
            run_id = os.path.basename(run_folder)
            
            # Upload to Firestore
            return self.upload_deployment_stats(stats, run_id)
            
        except Exception as e:
            print(f"[FIREBASE ERROR] Failed to upload run folder: {e}")
            return None
    
    def get_recent_deployments(self, limit=10):
        """
        Retrieve recent deployments from Firestore
        
        Args:
            limit: Maximum number of deployments to retrieve
        
        Returns:
            List of deployment documents
        """
        if not self.initialized:
            return []
        
        try:
            docs = self.db.collection('deployments').order_by('timestamp', direction='DESCENDING').limit(limit).stream()
            
            deployments = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                deployments.append(data)
            
            return deployments
            
        except Exception as e:
            print(f"[FIREBASE ERROR] Failed to retrieve deployments: {e}")
            return []
