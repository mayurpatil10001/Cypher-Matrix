#!/usr/bin/env python3
"""
Enhanced Advanced DeepFake Detection & Secure Browsing Platform v3.0
A comprehensive AI-powered solution for media authenticity verification and secure content analysis

Features:
- Advanced neural network detection with CNN-LSTM hybrid architecture
- Real-time video monitoring with multi-source support
- Multi-modal analysis fusion (video, audio, metadata)
- Comprehensive analytics dashboard with interactive visualizations
- Enhanced face manipulation detection with 8 analysis metrics
- Performance optimization with hardware detection
- Professional GUI with tabbed interface
- Database integration for analysis history
- Privacy protection with data anonymization
- Comprehensive logging and error handling

Author: AI Security Solutions Team
Version: 3.0 Enhanced
License: MIT
"""

import os
import sys
import subprocess
import pkg_resources
from packaging import version
import warnings
warnings.filterwarnings('ignore')

def check_and_install_dependencies():
    """Check and install required packages with user confirmation"""
    required_packages = [
        'opencv-python',
        'torch',
        'torchvision', 
        'torchaudio',
        'numpy',
        'pillow',
        'matplotlib',
        'seaborn',
        'scipy',
        'scikit-learn',
        'cryptography',
        'requests',
        'psutil',
        'selenium',
        'webdriver-manager',
        'pyotp' # Added for TOTP in AuthenticationManager
    ]
    
    try:
        installed_packages = [pkg.project_name.lower() for pkg in pkg_resources.working_set]
        missing_packages = []
        
        for package in required_packages:
            # Handle torch, torchvision, torchaudio as a group for checking
            if package in ['torch', 'torchvision', 'torchaudio']:
                if 'torch' not in installed_packages: # Just check for 'torch' as a representative
                    if 'torch' not in [m.lower() for m in missing_packages]: # Avoid duplicates
                        missing_packages.append('torch')
                continue # Skip individual checks for torchvision/torchaudio if torch is handled
            
            if package.lower() not in installed_packages:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"Missing packages detected: {missing_packages}")
            response = input("Install missing packages automatically? (y/n): ").lower().strip()
            
            if response == 'y':
                print("Installing required packages...")
                for package in missing_packages:
                    try:
                        print(f"Installing {package}...")
                        # Special handling for torch to install all three
                        if package == 'torch':
                            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--quiet"])
                            print(f"✓ Successfully installed torch, torchvision, torchaudio")
                        else:
                            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
                            print(f"✓ Successfully installed {package}")
                    except subprocess.CalledProcessError as e:
                        print(f"✗ Failed to install {package}: {e}")
                        return False
                print("All packages installed successfully!")
            else:
                print("Please install missing packages manually:")
                for package in missing_packages:
                    print(f"  pip install {package}")
                return False
                
        return True
        
    except Exception as e:
        print(f"Dependency check failed: {e}")
        print("Continuing with available packages...")
        return True

# Install dependencies if needed
print("Enhanced DeepFake Detection Platform v3.0")
print("=========================================")
print("Checking dependencies...")

if not check_and_install_dependencies():
    print("Some dependencies are missing. The application may not function properly.")
    input("Press Enter to continue anyway, or Ctrl+C to exit...")

# Import all required modules
print("Loading modules...")

try:
    import cv2
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    import torchvision.models as models  # Added for pre-trained models
    from torch.utils.data import DataLoader, Dataset
except ImportError as e:
    print(f"Critical import error: {e}")
    print("Please install PyTorch and OpenCV: pip install torch torchvision opencv-python")
    sys.exit(1)

import sqlite3
import hashlib
import json
import threading
import time
import requests
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import base64
from cryptography.fernet import Fernet
import tempfile
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import simpledialog  # Fixed: Imported simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    SEABORN_AVAILABLE = True
except ImportError:
    print("Seaborn not available, using matplotlib defaults")
    SEABORN_AVAILABLE = False

from PIL import Image, ImageTk
import io
import uuid
from collections import deque
from scipy import stats
import pickle
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import secrets
import wave
import struct

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    print("psutil not available - system monitoring will be limited")
    PSUTIL_AVAILABLE = False

# Try to import selenium with fallback
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    print("Selenium or webdriver-manager not available. Secure browsing features will be limited.")
    SELENIUM_AVAILABLE = False

# Configure comprehensive logging
def setup_logging():
    """Setup comprehensive logging system"""
    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # File handler
    file_handler = logging.FileHandler('logs/deepfake_platform.log')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler  
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    console_handler.setLevel(logging.INFO)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)

logger = setup_logging()
logger.info("Enhanced DeepFake Detection Platform v3.0 starting...")

class AdvancedDeepFakeDetector(nn.Module):
    """
    Advanced CNN-LSTM hybrid model for deepfake detection
    
    Architecture:
    - Pre-trained EfficientNet-B0 feature extractor (improved for better detection)
    - Multi-head self-attention mechanism
    - Bidirectional LSTM for temporal analysis
    - Multi-layer classification head with dropout
    """
    
    def __init__(self, input_channels=3, hidden_size=512, num_classes=2):
        super(AdvancedDeepFakeDetector, self).__init__()
        
        logger.info(f"Initializing AdvancedDeepFakeDetector with {num_classes} classes")
        
        # Use pre-trained EfficientNet-B0 for better feature extraction
        self.feature_extractor = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # Replace classifier to output 512 features
        self.feature_extractor.classifier = nn.Linear(self.feature_extractor.classifier[1].in_features, 512)
        
        # Multi-head self-attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=512, 
            num_heads=8, 
            dropout=0.1,
            batch_first=True
        )
        
        # Bidirectional LSTM for temporal analysis
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Classification head with dropout and batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights using best practices
        self._initialize_weights()
        
        logger.info("AdvancedDeepFakeDetector initialized with pre-trained EfficientNet backbone")
        
    def _initialize_weights(self):
        """Initialize model weights using Xavier/He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, channels, height, width]
            
        Returns:
            output: Classification logits [batch_size, num_classes]
            attention_weights: Attention weights for interpretability
        """
        batch_size, seq_len, c, h, w = x.size()
        
        # Process each frame through pre-trained feature extractor
        features = []
        for i in range(seq_len):
            frame_features = self.feature_extractor(x[:, i])
            features.append(frame_features)
        
        # Stack features for sequence processing
        features = torch.stack(features, dim=1)  # [batch_size, seq_len, 512]
        
        # Apply multi-head self-attention
        attended_features, attention_weights = self.attention(features, features, features)
        
        # Add residual connection
        attended_features = attended_features + features
        
        # LSTM processing for temporal analysis
        lstm_out, (hidden, cell) = self.lstm(attended_features)
        
        # Use the last time step output for classification
        final_features = lstm_out[:, -1, :]
        
        # Classification
        output = self.classifier(final_features)
        
        return output, attention_weights

class EnhancedFaceAnalyzer:
    """
    Enhanced face analysis system with 8 comprehensive detection methods:
    1. Compression artifact detection
    2. Blending artifact detection  
    3. Lighting consistency analysis
    4. Texture naturalness analysis
    5. Geometric consistency analysis
    6. Color naturalness analysis
    7. Edge consistency analysis
    8. Frequency domain analysis
    """
    
    def __init__(self):
        logger.info("Initializing EnhancedFaceAnalyzer")
        
        # Initialize OpenCV cascade classifiers
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            self.profile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_profileface.xml'
            )
            logger.info("OpenCV cascade classifiers loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load cascade classifiers: {e}")
            self.face_cascade = None
            self.eye_cascade = None
            self.profile_cascade = None
        
        # Initialize feature detectors
        try:
            self.sift = cv2.SIFT_create(nfeatures=500)
            self.orb = cv2.ORB_create(nfeatures=500)
            logger.info("Feature detectors initialized")
        except Exception as e:
            logger.warning(f"Feature detectors not available: {e}")
            self.sift = None
            self.orb = None
            
        # Analysis thresholds and parameters
        self.analysis_params = {
            'compression_threshold': 0.6,
            'blending_threshold': 0.5,
            'lighting_threshold': 0.4,
            'texture_threshold': 0.5,
            'geometry_threshold': 0.3,
            'color_threshold': 0.4,
            'edge_threshold': 0.5,
            'frequency_threshold': 0.4
        }
        
        logger.info("EnhancedFaceAnalyzer initialized successfully")
    
    def comprehensive_face_analysis(self, frame, return_visualizations=False):
        """
        Perform comprehensive face analysis with multiple detection methods
        
        Args:
            frame: Input image frame
            return_visualizations: Whether to return visualization overlays
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        try:
            analysis_start = time.time()
            
            results = {
                'faces_detected': 0,
                'face_regions': [],
                'analysis_results': [],
                'overall_confidence': 0.0,
                'processing_time': 0.0,
                'timestamp': datetime.now().isoformat(),
                'analysis_methods': [
                    'compression_artifacts',
                    'blending_artifacts', 
                    'lighting_consistency',
                    'texture_naturalness',
                    'geometric_consistency',
                    'color_naturalness',
                    'edge_consistency',
                    'frequency_analysis'
                ],
                'visualizations': {} if return_visualizations else None
            }
            
            if self.face_cascade is None:
                logger.warning("Face cascade not available, using center region analysis")
                # Analyze center region as fallback
                h, w = frame.shape[:2]
                center_region = (w//4, h//4, w//2, h//2)
                results['face_regions'] = [{'id': 0, 'bbox': center_region, 'type': 'center_region'}]
                results['faces_detected'] = 1
            else:
                # Convert to grayscale for detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Multi-scale face detection with multiple cascades
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5,
                    minSize=(50, 50),
                    maxSize=(500, 500),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # Try profile detection if no frontal faces found
                if len(faces) == 0 and self.profile_cascade is not None:
                    faces = self.profile_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(50, 50)
                    )
                
                results['faces_detected'] = len(faces)
                
                # Store face region information
                for i, (x, y, w, h) in enumerate(faces):
                    results['face_regions'].append({
                        'id': i,
                        'bbox': (x, y, w, h),
                        'size': w * h,
                        'aspect_ratio': w / h,
                        'center': (x + w//2, y + h//2),
                        'type': 'detected_face'
                    })
            
            # Analyze each detected face region
            total_suspicion = 0.0
            visualized_frame = frame.copy() if return_visualizations else None
            
            for i, face_info in enumerate(results['face_regions']):
                x, y, w, h = face_info['bbox']
                
                # Extract face region
                face_region = frame[y:y+h, x:x+w]
                face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                
                # Perform comprehensive analysis
                analysis = self._analyze_single_face(face_region, face_gray)
                analysis.update({
                    'face_id': i,
                    'bbox': (x, y, w, h),
                    'face_info': face_info
                })
                
                results['analysis_results'].append(analysis)
                total_suspicion += analysis.get('overall_suspicion', 0.0)
                
                # Add visualization if requested
                if return_visualizations and visualized_frame is not None:
                    self._add_face_visualization(visualized_frame, (x, y, w, h), analysis)
            
            # Calculate overall confidence
            if len(results['face_regions']) > 0:
                results['overall_confidence'] = total_suspicion / len(results['face_regions'])
            
            results['processing_time'] = time.time() - analysis_start
            
            # Store visualizations
            if return_visualizations:
                results['visualizations']['annotated_frame'] = visualized_frame
                results['visualizations']['detection_overlay'] = self._create_detection_overlay(frame, results)
            
            logger.debug(f"Face analysis completed in {results['processing_time']:.3f}s, "
                        f"detected {results['faces_detected']} faces with "
                        f"{results['overall_confidence']:.1%} overall suspicion")
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive face analysis failed: {e}")
            return {
                'error': str(e),
                'faces_detected': 0,
                'overall_confidence': 0.0,
                'processing_time': time.time() - analysis_start if 'analysis_start' in locals() else 0.0,
                'timestamp': datetime.now().isoformat()
            }
    
    def _analyze_single_face(self, face_region, face_gray):
        """
        Analyze a single face region using all 8 detection methods
        
        Args:
            face_region: Color face region (BGR)
            face_gray: Grayscale face region
            
        Returns:
            Dictionary with analysis results for all methods
        """
        try:
            analysis = {
                'compression_artifacts': 0.0,
                'blending_artifacts': 0.0, 
                'lighting_consistency': 0.0,
                'texture_naturalness': 0.0,
                'geometric_consistency': 0.0,
                'color_naturalness': 0.0,
                'edge_consistency': 0.0,
                'frequency_analysis': 0.0,
                'overall_suspicion': 0.0,
                'analysis_details': {},
                'confidence_scores': {}
            }
            
            # Method 1: Compression artifact detection using DCT analysis
            analysis['compression_artifacts'] = self._detect_compression_artifacts(face_region)
            
            # Method 2: Blending artifact detection using edge analysis
            analysis['blending_artifacts'] = self._detect_blending_artifacts(face_region)
            
            # Method 3: Lighting consistency analysis
            analysis['lighting_consistency'] = self._analyze_lighting_consistency(face_region)
            
            # Method 4: Texture naturalness using LBP analysis
            analysis['texture_naturalness'] = self._analyze_texture_naturalness(face_gray)
            
            # Method 5: Geometric consistency analysis
            analysis['geometric_consistency'] = self._analyze_geometric_consistency(face_gray)
            
            # Method 6: Color naturalness analysis
            analysis['color_naturalness'] = self._analyze_color_naturalness(face_region)
            
            # Method 7: Edge consistency analysis
            analysis['edge_consistency'] = self._analyze_edge_consistency(face_gray)
            
            # Method 8: Frequency domain analysis
            analysis['frequency_analysis'] = self._frequency_domain_analysis(face_gray)
            
            # Calculate overall suspicion score with weighted average
            weights = {
                'compression_artifacts': 0.15,
                'blending_artifacts': 0.15,
                'lighting_consistency': 0.12,
                'texture_naturalness': 0.15,
                'geometric_consistency': 0.10,
                'color_naturalness': 0.13,
                'edge_consistency': 0.10,
                'frequency_analysis': 0.10
            }
            
            weighted_score = 0.0
            for method, weight in weights.items():
                score = analysis.get(method, 0.0)
                weighted_score += score * weight
                analysis['confidence_scores'][method] = score
            
            analysis['overall_suspicion'] = min(max(weighted_score, 0.0), 1.0)
            
            # Store analysis details
            analysis['analysis_details'] = {
                'region_size': face_region.shape,
                'analysis_weights': weights,
                'suspicious_methods': [
                    method for method, score in analysis['confidence_scores'].items() 
                    if score > self.analysis_params.get(f"{method.split('_')[0]}_threshold", 0.5)
                ]
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Single face analysis failed: {e}")
            return {
                'error': str(e),
                'overall_suspicion': 0.0,
                'analysis_details': {'error': str(e)}
            }
    
    def _detect_compression_artifacts(self, face_region):
        """
        Detect compression artifacts using DCT (Discrete Cosine Transform) analysis
        
        Compression artifacts appear as:
        - Blocking effects in 8x8 DCT blocks
        - Reduced high-frequency content
        - Quantization noise patterns
        """
        try:
            # Convert to grayscale and standardize size
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Resize to multiple of 8 for DCT analysis (JPEG uses 8x8 blocks)
            target_size = 128  # 128x128 = 16x16 blocks of 8x8
            gray_face = cv2.resize(gray_face, (target_size, target_size))
            
            # Apply DCT to the entire image
            dct_full = cv2.dct(np.float32(gray_face))
            
            # Analyze DCT coefficient distribution
            # Split into frequency bands
            block_size = 8
            compression_scores = []
            
            # Analyze each 8x8 block
            for i in range(0, target_size, block_size):
                for j in range(0, target_size, block_size):
                    block = dct_full[i:i+block_size, j:j+block_size]
                    
                    # Calculate energy distribution
                    # DC component (top-left)
                    dc_energy = abs(block[0, 0])
                    
                    # Low frequency (top-left 3x3)
                    low_freq_energy = np.sum(np.abs(block[:3, :3])) - dc_energy
                    
                    # High frequency (remaining coefficients)
                    high_freq_energy = np.sum(np.abs(block)) - dc_energy - low_freq_energy
                    
                    total_energy = dc_energy + low_freq_energy + high_freq_energy
                    
                    if total_energy > 0:
                        # Compression typically reduces high-frequency content
                        high_freq_ratio = high_freq_energy / total_energy
                        
                        # Normal images have moderate high-frequency content (0.1-0.3)
                        # Heavily compressed images have very low high-frequency content (<0.1)
                        if high_freq_ratio < 0.05:  # Very low high frequencies
                            compression_scores.append(0.8)
                        elif high_freq_ratio < 0.1:  # Low high frequencies  
                            compression_scores.append(0.6)
                        elif high_freq_ratio > 0.5:  # Unusually high (possible artifact)
                            compression_scores.append(0.4)
                        else:
                            compression_scores.append(0.1)
                    else:
                        compression_scores.append(0.0)
            
            # Average compression artifact score
            avg_compression_score = np.mean(compression_scores) if compression_scores else 0.0
            
            # Additional analysis: Look for blocking artifacts
            blocking_score = self._detect_blocking_artifacts(gray_face, block_size)
            
            # Combine scores
            final_score = (avg_compression_score * 0.7 + blocking_score * 0.3)
            
            return min(max(final_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Compression artifact detection failed: {e}")
            return 0.0
    
    def _detect_blocking_artifacts(self, image, block_size=8):
        """Detect JPEG-like blocking artifacts"""
        try:
            h, w = image.shape
            blocking_scores = []
            
            # Check horizontal block boundaries
            for i in range(block_size, h, block_size):
                if i < h - 1:
                    # Calculate difference across block boundary
                    row_above = image[i-1, :]
                    row_below = image[i, :]
                    boundary_diff = np.mean(np.abs(row_above.astype(float) - row_below.astype(float)))
                    
                    # Calculate difference within blocks
                    within_diff_above = np.mean(np.abs(np.diff(row_above.astype(float))))
                    within_diff_below = np.mean(np.abs(np.diff(row_below.astype(float))))
                    within_diff = (within_diff_above + within_diff_below) / 2
                    
                    # Blocking artifacts show high boundary differences relative to within-block differences
                    if within_diff > 0:
                        blocking_ratio = boundary_diff / within_diff
                        blocking_scores.append(min(blocking_ratio / 3.0, 1.0))  # Normalize
            
            # Check vertical block boundaries  
            for j in range(block_size, w, block_size):
                if j < w - 1:
                    col_left = image[:, j-1]
                    col_right = image[:, j]
                    boundary_diff = np.mean(np.abs(col_left.astype(float) - col_right.astype(float)))
                    
                    within_diff_left = np.mean(np.abs(np.diff(col_left.astype(float))))
                    within_diff_right = np.mean(np.abs(np.diff(col_right.astype(float))))
                    within_diff = (within_diff_left + within_diff_right) / 2
                    
                    if within_diff > 0:
                        blocking_ratio = boundary_diff / within_diff
                        blocking_scores.append(min(blocking_ratio / 3.0, 1.0))
            
            return np.mean(blocking_scores) if blocking_scores else 0.0
            
        except Exception as e:
            logger.error(f"Blocking artifact detection failed: {e}")
            return 0.0
    
    def _detect_blending_artifacts(self, face_region):
        """
        Detect blending artifacts using multi-scale edge analysis
        
        Blending artifacts appear as:
        - Inconsistent edge patterns
        - Unnatural smoothing
        - Sharp transitions at blend boundaries
        """
        try:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray_face, (3, 3), 0)
            
            # Multi-scale edge detection
            edge_maps = []
            
            # Canny edge detection with different thresholds
            for low_thresh in [30, 50, 70]:
                high_thresh = low_thresh * 2
                edges = cv2.Canny(blurred, low_thresh, high_thresh)
                edge_maps.append(edges)
            
            # Sobel edge detection
            sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            edge_maps.append((sobel_magnitude > np.percentile(sobel_magnitude, 70)).astype(np.uint8) * 255)
            
            # Laplacian edge detection
            laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
            edge_maps.append((np.abs(laplacian) > np.percentile(np.abs(laplacian), 70)).astype(np.uint8) * 255)
            
            # Analyze edge consistency across different methods
            inconsistency_scores = []
            
            for i in range(len(edge_maps)):
                for j in range(i+1, len(edge_maps)):
                    # Calculate difference between edge maps
                    diff = cv2.absdiff(edge_maps[i], edge_maps[j])
                    inconsistency = np.mean(diff) / 255.0
                    inconsistency_scores.append(inconsistency)
            
            # High inconsistency indicates possible blending artifacts
            avg_inconsistency = np.mean(inconsistency_scores) if inconsistency_scores else 0.0
            
            # Analyze edge sharpness distribution
            edge_sharpness = cv2.Sobel(gray_face, cv2.CV_64F, 1, 1, ksize=3)
            sharpness_var = np.var(np.abs(edge_sharpness))
            sharpness_score = 1.0 - min(sharpness_var / 10000.0, 1.0)  # Low variance indicates unnatural smoothing
            
            # Combine scores
            final_score = (avg_inconsistency * 0.6 + sharpness_score * 0.4)
            
            return min(max(final_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Blending artifact detection failed: {e}")
            return 0.0
    
    def _analyze_lighting_consistency(self, face_region):
        """Analyze lighting consistency across face"""
        try:
            # Convert to HSV for value channel
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            value = hsv[:,:,2]
            
            # Calculate gradient magnitude
            grad_x = cv2.Sobel(value, cv2.CV_64F, 1, 0, ksize=5)
            grad_y = cv2.Sobel(value, cv2.CV_64F, 0, 1, ksize=5)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # High gradient variation indicates inconsistent lighting
            grad_var = np.var(grad_mag)
            lighting_score = min(grad_var / 5000.0, 1.0)
            
            # Analyze shadow consistency
            _, binary = cv2.threshold(value, 0, 255, cv2.THRESH_OTSU)
            shadow_ratio = np.sum(binary == 0) / binary.size
            shadow_score = abs(shadow_ratio - 0.2) * 5.0  # Assume ~20% shadow in natural faces
            
            final_score = (lighting_score * 0.7 + shadow_score * 0.3)
            return min(max(final_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Lighting consistency analysis failed: {e}")
            return 0.0
    
    def _analyze_texture_naturalness(self, face_gray):
        """Analyze texture using Local Binary Patterns (LBP)"""
        try:
            # Compute LBP
            radius = 3
            n_points = 8 * radius
            lbp = np.zeros_like(face_gray)
            for i in range(radius, face_gray.shape[0] - radius):
                for j in range(radius, face_gray.shape[1] - radius):
                    center = face_gray[i, j]
                    code = 0
                    for k in range(8):
                        x = i + int(round(radius * np.sin(2 * np.pi * k / 8)))
                        y = j - int(round(radius * np.cos(2 * np.pi * k / 8)))
                        code |= (face_gray[x, y] > center) << k
                    lbp[i, j] = code
            
            # Histogram analysis
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 257), density=True)
            
            # Natural textures have more uniform distribution
            entropy = stats.entropy(hist + 1e-10)  # Add epsilon to avoid log(0)
            max_entropy = np.log2(256)  # Maximum for 8-bit LBP
            texture_score = 1.0 - (entropy / max_entropy)  # Low entropy = unnatural
            
            return min(max(texture_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Texture naturalness analysis failed: {e}")
            return 0.0
    
    def _analyze_geometric_consistency(self, face_gray):
        """Analyze geometric consistency using landmark ratios"""
        try:
            if self.eye_cascade is None or self.face_cascade is None:
                return 0.0
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20,20))
            
            if len(eyes) != 2:
                return 0.5  # Uncertain if not exactly two eyes
            
            # Calculate eye distance ratio
            eye1, eye2 = sorted(eyes, key=lambda e: e[0])
            eye_dist = abs(eye1[0] + eye1[2]/2 - (eye2[0] + eye2[2]/2))
            face_width = face_gray.shape[1]
            ratio = eye_dist / face_width
            
            # Natural ratio is ~0.45-0.6
            geometry_score = abs(ratio - 0.525) * 5.0
            
            # Add symmetry analysis
            left_half = face_gray[:, :face_gray.shape[1]//2]
            right_half = cv2.flip(face_gray[:, face_gray.shape[1]//2:], 1)
            symmetry_diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0
            
            final_score = (geometry_score * 0.4 + symmetry_diff * 0.6)
            return min(max(final_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Geometric consistency analysis failed: {e}")
            return 0.0
    
    def _analyze_color_naturalness(self, face_region):
        """Analyze color distribution naturalness"""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
            
            # Analyze A and B channels (color information)
            a_channel = lab[:,:,1]
            b_channel = lab[:,:,2]
            
            # Natural skin tones have specific distributions
            a_mean = np.mean(a_channel)
            b_mean = np.mean(b_channel)
            color_score = abs(a_mean - 130) / 20.0 + abs(b_mean - 140) / 20.0  # Approximate natural skin means
            
            # Color histogram entropy
            hist_a, _ = np.histogram(a_channel.ravel(), bins=256, density=True)
            hist_b, _ = np.histogram(b_channel.ravel(), bins=256, density=True)
            entropy_a = stats.entropy(hist_a + 1e-10)
            entropy_b = stats.entropy(hist_b + 1e-10)
            entropy_score = 2.0 - ((entropy_a + entropy_b) / (2 * np.log2(256)))  # Low entropy = unnatural
            
            final_score = (color_score * 0.5 + entropy_score * 0.5) / 2.0
            return min(max(final_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Color naturalness analysis failed: {e}")
            return 0.0
    
    def _analyze_edge_consistency(self, face_gray):
        """Analyze edge consistency using Hough transform"""
        try:
            edges = cv2.Canny(face_gray, 50, 150)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=5)
            
            if lines is None:
                return 0.0
            
            # Analyze line orientations
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
            
            # High variance in angles indicates natural edges
            angle_var = np.var(angles) if angles else 0.0
            consistency_score = 1.0 - min(angle_var / 5000.0, 1.0)  # Low variance = artificial
            
            # Edge density
            edge_density = np.sum(edges > 0) / edges.size
            density_score = abs(edge_density - 0.15) * 6.67  # Natural faces ~15% edges
            
            final_score = (consistency_score * 0.6 + density_score * 0.4)
            return min(max(final_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Edge consistency analysis failed: {e}")
            return 0.0
    
    def _frequency_domain_analysis(self, face_gray):
        """Frequency domain analysis using FFT"""
        try:
            # Apply FFT
            f = np.fft.fft2(face_gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-10)
            
            # Analyze high frequency content
            h, w = magnitude_spectrum.shape
            center_h, center_w = h//2, w//2
            
            # Mask center (low frequencies)
            mask = np.ones((h, w), np.uint8)
            cv2.circle(mask, (center_w, center_h), 30, 0, -1)  # Remove low freq
            
            high_freq = magnitude_spectrum * mask
            high_freq_energy = np.sum(high_freq)
            
            total_energy = np.sum(magnitude_spectrum)
            high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0.0
            
            # Deepfakes often have reduced high frequencies
            freq_score = 1.0 - min(high_freq_ratio / 0.3, 1.0)  # Assume natural >30% high freq
            
            # Look for periodic patterns (possible artifacts)
            autocorr = np.abs(np.fft.ifft2(np.abs(f)**2))
            peak_ratio = np.max(autocorr[1:]) / autocorr[0,0]
            pattern_score = min(peak_ratio * 10.0, 1.0)
            
            final_score = (freq_score * 0.7 + pattern_score * 0.3)
            return min(max(final_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Frequency domain analysis failed: {e}")
            return 0.0
    
    def _add_face_visualization(self, frame, bbox, analysis):
        """Add visualization overlays to frame"""
        try:
            x, y, w, h = bbox
            suspicion = analysis['overall_suspicion']
            
            # Draw bounding box with color based on suspicion
            color = (0, int(255 * (1 - suspicion)), int(255 * suspicion))  # Green to Red
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Add suspicion score
            cv2.putText(frame, f"{suspicion:.1%}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Add suspicious methods icons or text
            suspicious = analysis['analysis_details']['suspicious_methods']
            if suspicious:
                warn_text = "Warnings: " + ", ".join([m[0].upper() for m in suspicious])
                cv2.putText(frame, warn_text, (x, y+h+20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
        except Exception as e:
            logger.warning(f"Visualization addition failed: {e}")
    
    def _create_detection_overlay(self, frame, results):
        """Create separate overlay image for detections"""
        try:
            overlay = np.zeros_like(frame)
            
            for analysis in results['analysis_results']:
                x, y, w, h = analysis['bbox']
                suspicion = analysis['overall_suspicion']
                color = (0, int(255 * (1 - suspicion)), int(255 * suspicion))
                
                # Draw semi-transparent rectangle
                cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
            
            # Blend overlay with original
            blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            return blended
            
        except Exception as e:
            logger.warning(f"Detection overlay creation failed: {e}")
            return frame

class DatabaseManager:
    """
    Enhanced database manager for analysis history and known samples
    with privacy protection
    """
    def __init__(self, db_path='deepfake_analysis.db'):
        self.db_path = db_path
        self._initialize_database()
        logger.info(f"Database initialized at {db_path}")
    
    def _initialize_database(self):
        """Initialize SQLite database with enhanced schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Analysis results table with anonymized data
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS analysis_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        analysis_id TEXT UNIQUE,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        media_type TEXT,
                        confidence REAL,
                        is_deepfake BOOLEAN,
                        processing_time REAL,
                        model_version TEXT,
                        face_analysis BLOB,  -- Pickled anonymized data
                        neural_network_result BLOB,  -- Pickled data
                        file_hash TEXT UNIQUE,
                        anonymized_metadata BLOB
                    )
                ''')
                
                # Known samples table for quick lookup
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS known_samples (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_hash TEXT UNIQUE,
                        label TEXT,  -- 'real' or 'fake'
                        description TEXT,
                        added_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        source TEXT
                    )
                ''')
                
                # System metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        cpu_usage REAL,
                        memory_usage REAL,
                        gpu_usage REAL,
                        analysis_count INTEGER,
                        detection_rate REAL
                    )
                ''')
                
                # User table for authentication
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        salt TEXT NOT NULL,
                        role TEXT DEFAULT 'user',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        last_login DATETIME,
                        failed_attempts INTEGER DEFAULT 0,
                        locked_until DATETIME,
                        totp_secret TEXT
                    )
                ''')
                
                # User sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        session_token TEXT UNIQUE NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        expires_at DATETIME NOT NULL,
                        last_activity DATETIME,
                        ip_address TEXT,
                        is_active BOOLEAN DEFAULT 1,
                        FOREIGN KEY (user_id) REFERENCES users(id)
                    )
                ''')
                
                # Login attempts log
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS login_attempts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        ip_address TEXT,
                        success BOOLEAN NOT NULL,
                        failure_reason TEXT
                    )
                ''')
                
                conn.commit()
                logger.info("Database schema initialized successfully")
                
        except sqlite3.Error as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def store_analysis_result(self, result: Dict):
        """Store analysis result with data anonymization"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Generate unique analysis ID
                analysis_id = str(uuid.uuid4())
                
                # Anonymize sensitive data
                anonymized_result = self._anonymize_data(result)
                
                # Compute file hash for deduplication
                file_hash = self._compute_file_hash(result.get('file_path'))
                
                # Pickle complex data structures
                face_analysis_pickle = pickle.dumps(anonymized_result.get('face_analysis'))
                nn_result_pickle = pickle.dumps(anonymized_result.get('neural_network_result'))
                metadata_pickle = pickle.dumps(anonymized_result.get('metadata', {}))
                
                cursor.execute('''
                    INSERT OR REPLACE INTO analysis_results 
                    (analysis_id, media_type, confidence, is_deepfake, 
                     processing_time, model_version, face_analysis, 
                     neural_network_result, file_hash, anonymized_metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    analysis_id,
                    result.get('media_type'),
                    result.get('confidence'),
                    result.get('is_deepfake'),
                    result.get('processing_time'),
                    result.get('model_version'),
                    face_analysis_pickle,
                    nn_result_pickle,
                    file_hash,
                    metadata_pickle
                ))
                
                conn.commit()
                logger.debug(f"Stored analysis result {analysis_id}")
                
                return analysis_id
                
        except sqlite3.Error as e:
            logger.error(f"Failed to store analysis result: {e}")
            return None
    
    def _anonymize_data(self, data: Dict) -> Dict:
        """Anonymize sensitive information in analysis data"""
        anonymized = data.copy()
        
        # Remove file paths
        if 'file_path' in anonymized:
            del anonymized['file_path']
        
        # Anonymize metadata if present
        if 'metadata' in anonymized:
            metadata = anonymized['metadata']
            sensitive_keys = ['gps', 'location', 'device_id', 'user', 'author']
            for key in sensitive_keys:
                if key in metadata:
                    metadata[key] = '[ANONYMIZED]'
        
        # Hash any identifiers
        if 'session_id' in anonymized:
            anonymized['session_id'] = hashlib.sha256(anonymized['session_id'].encode()).hexdigest()
        
        return anonymized
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file content"""
        if not file_path or not os.path.exists(file_path):
            return None
        
        hash_obj = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(4096):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    
    def check_known_sample(self, file_path: str) -> Optional[str]:
        """Check if file is a known sample"""
        try:
            file_hash = self._compute_file_hash(file_path)
            if not file_hash:
                return None
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT label FROM known_samples 
                    WHERE file_hash = ?
                ''', (file_hash,))
                result = cursor.fetchone()
                
                return result[0] if result else None
                
        except sqlite3.Error as e:
            logger.error(f"Known sample check failed: {e}")
            return None
    
    def add_known_sample(self, file_path: str, label: str, description: str = '', source: str = ''):
        """Add known sample to database"""
        try:
            file_hash = self._compute_file_hash(file_path)
            if not file_hash:
                return False
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO known_samples 
                    (file_hash, label, description, source)
                    VALUES (?, ?, ?, ?)
                ''', (file_hash, label.lower(), description, source))
                conn.commit()
                
                logger.info(f"Added known sample {file_hash} as {label}")
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Failed to add known sample: {e}")
            return False
    
    def store_system_metrics(self, metrics: Dict):
        """Store system performance metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO system_metrics 
                    (cpu_usage, memory_usage, gpu_usage, 
                     analysis_count, detection_rate)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    metrics.get('cpu_usage', 0.0),
                    metrics.get('memory_usage', 0.0),
                    metrics.get('gpu_usage', 0.0),
                    metrics.get('analysis_count', 0),
                    metrics.get('detection_rate', 0.0)
                ))
                conn.commit()
                
        except sqlite3.Error as e:
            logger.error(f"Failed to store system metrics: {e}")
    
    def get_recent_analyses(self, limit: int = 100) -> List[Dict]:
        """Retrieve recent analysis results"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM analysis_results 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
                rows = cursor.fetchall()
                
                results = []
                for row in rows:
                    result = dict(row)
                    # Unpickle complex fields
                    if result['face_analysis']:
                        result['face_analysis'] = pickle.loads(result['face_analysis'])
                    if result['neural_network_result']:
                        result['neural_network_result'] = pickle.loads(result['neural_network_result'])
                    if result['anonymized_metadata']:
                        result['anonymized_metadata'] = pickle.loads(result['anonymized_metadata'])
                    results.append(result)
                
                return results
                
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve recent analyses: {e}")
            return []
    
    def get_detection_trend(self, days: int = 30) -> List[Dict]:
        """Get detection trend over time"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT timestamp, confidence 
                    FROM analysis_results 
                    WHERE timestamp >= datetime('now', ?)
                    ORDER BY timestamp ASC
                ''', (f'-{days} days',))
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except sqlite3.Error as e:
            logger.error(f"Failed to get detection trend: {e}")
            return []
    
    def cleanup_old_data(self, retention_days: int = 90):
        """Cleanup old data while maintaining privacy"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Cleanup old analyses
                cursor.execute('''
                    DELETE FROM analysis_results 
                    WHERE timestamp < datetime('now', ?)
                ''', (f'-{retention_days} days',))
                
                # Cleanup old metrics
                cursor.execute('''
                    DELETE FROM system_metrics 
                    WHERE timestamp < datetime('now', ?)
                ''', (f'-{retention_days} days',))
                
                # Cleanup expired sessions
                cursor.execute('''
                    DELETE FROM user_sessions 
                    WHERE expires_at < CURRENT_TIMESTAMP
                ''')
                
                # Cleanup old login attempts
                cursor.execute('''
                    DELETE FROM login_attempts 
                    WHERE timestamp < datetime('now', ?)
                ''', (f'-{retention_days} days',))
                
                conn.commit()
                
                logger.info(f"Cleaned up data older than {retention_days} days")
                
        except sqlite3.Error as e:
            logger.error(f"Data cleanup failed: {e}")

class RealTimeAnalyticsEngine:
    """
    Real-time analytics engine for detection metrics and trends
    with interactive dashboard support
    """
    def __init__(self, database: DatabaseManager):
        self.database = database
        self.metrics_history = deque(maxlen=1000)  # Recent metrics buffer
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self._train_anomaly_detector()
        logger.info("RealTimeAnalyticsEngine initialized")
    
    def _train_anomaly_detector(self):
        """Train anomaly detector on historical data"""
        try:
            historical = self.database.get_recent_analyses(1000)
            if len(historical) > 50:
                features = np.array([[r['confidence'], r['processing_time']] for r in historical])
                scaled = self.scaler.fit_transform(features)
                self.anomaly_detector.fit(scaled)
                logger.info("Anomaly detector trained on historical data")
            else:
                logger.warning("Insufficient data for anomaly detector training")
                
        except Exception as e:
            logger.error(f"Anomaly detector training failed: {e}")
    
    def update_metrics(self, analysis_result: Dict):
        """Update analytics metrics in real-time"""
        try:
            # Add to history buffer
            self.metrics_history.append(analysis_result)
            
            # Detect anomalies
            features = np.array([[analysis_result['confidence'], analysis_result['processing_time']]])
            scaled = self.scaler.transform(features)
            is_anomaly = self.anomaly_detector.predict(scaled)[0] == -1
            
            if is_anomaly:
                logger.warning(f"Anomalous analysis detected: {analysis_result['analysis_id']}")
                # Could trigger alert
            
            # Store system metrics if psutil available
            if PSUTIL_AVAILABLE:
                system_metrics = {
                    'cpu_usage': psutil.cpu_percent(),
                    'memory_usage': psutil.virtual_memory().percent,
                    'gpu_usage': self._get_gpu_usage(),
                    'analysis_count': len(self.metrics_history),
                    'detection_rate': self._calculate_detection_rate()
                }
                self.database.store_system_metrics(system_metrics)
                
        except Exception as e:
            logger.error(f"Metrics update failed: {e}")
    
    def _get_gpu_usage(self) -> float:
        """Get GPU usage if available"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.utilization(0)
            return 0.0
        except:
            return 0.0
    
    def _calculate_detection_rate(self) -> float:
        """Calculate current detection rate"""
        if not self.metrics_history:
            return 0.0
        detections = sum(1 for r in self.metrics_history if r['is_deepfake'])
        return detections / len(self.metrics_history)
    
    def generate_analytics_report(self) -> Dict:
        """Generate comprehensive analytics report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'detection_statistics': {
                    'total_analyses': len(self.metrics_history),
                    'detection_count': sum(1 for r in self.metrics_history if r['is_deepfake']),
                    'average_confidence': np.mean([r['confidence'] for r in self.metrics_history]),
                    'average_processing_time': np.mean([r['processing_time'] for r in self.metrics_history]),
                    'media_types': {}
                },
                'system_performance': {
                    'cpu_usage': psutil.cpu_percent() if PSUTIL_AVAILABLE else None,
                    'memory_usage': psutil.virtual_memory().percent if PSUTIL_AVAILABLE else None,
                    'gpu_usage': self._get_gpu_usage()
                },
                'detection_trend': self.database.get_detection_trend(),
                'anomaly_detections': sum(1 for _ in range(len(self.metrics_history)) if self._is_anomaly(self.metrics_history[-1]))  # Placeholder
            }
            
            # Count media types
            for result in self.metrics_history:
                mtype = result.get('media_type', 'unknown')
                report['detection_statistics']['media_types'][mtype] = report['detection_statistics']['media_types'].get(mtype, 0) + 1
            
            logger.info("Generated analytics report")
            return report
            
        except Exception as e:
            logger.error(f"Analytics report generation failed: {e}")
            return {}
    
    def _is_anomaly(self, result: Dict) -> bool:
        """Check if analysis result is anomalous"""
        try:
            features = np.array([[result['confidence'], result['processing_time']]])
            scaled = self.scaler.transform(features)
            return self.anomaly_detector.predict(scaled)[0] == -1
        except:
            return False
    
    def generate_visualization_data(self) -> Dict:
        """Generate data for dashboard visualizations"""
        try:
            recent = list(self.metrics_history)[-100:]
            vis_data = {
                'confidence_histogram': np.histogram([r['confidence'] for r in recent], bins=10)[0].tolist(),
                'processing_time_trend': [r['processing_time'] for r in recent],
                'detection_pie': [
                    sum(1 for r in recent if r['is_deepfake']),
                    sum(1 for r in recent if not r['is_deepfake'])
                ],
                'timestamps': [r.get('timestamp') for r in recent]
            }
            return vis_data
            
        except Exception as e:
            logger.error(f"Visualization data generation failed: {e}")
            return {}

class SecureBrowserManager:
    """
    Secure browser manager for isolated content viewing
    with risk assessment and privacy protection
    """
    def __init__(self, database: Optional[DatabaseManager] = None):
        self.database = database
        self.active_sessions = {}
        self.security_config = {
            'trusted_domains': ['youtube.com', 'vimeo.com', 'example.com'],  # Add trusted domains
            'risk_threshold': 0.5,
            'session_timeout_minutes': 30,
            'max_sessions': 5
        }
        if SELENIUM_AVAILABLE:
            logger.info("Secure browsing capabilities enabled")
        else:
            logger.warning("Secure browsing disabled - Selenium not available")
    
    def create_secure_session(self, url: str, incognito: bool = False) -> Dict:
        """Create secure browsing session with risk assessment"""
        if not SELENIUM_AVAILABLE:
            return {'success': False, 'error': 'Secure browsing not available'}
        
        try:
            # Check active sessions limit
            if len(self.active_sessions) >= self.security_config['max_sessions']:
                return {'success': False, 'error': 'Maximum sessions reached'}
            
            # Generate session ID
            session_id = str(uuid.uuid4())
            
            # Setup browser options
            options = Options()
            
            if not incognito:
                options.add_argument("--headless")
                options.add_argument("--disable-gpu")
            else:
                options.add_argument("--incognito")
                # For visible incognito, don't set headless
            
            # Security hardening
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-plugins")
            options.add_argument("--no-sandbox")  # Careful with this in production
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-infobars")
            options.add_argument("--disable-notifications")
            options.add_argument("--mute-audio")  # Mute by default
            
            # Create temporary profile for isolation
            temp_profile_dir = tempfile.mkdtemp(prefix='secure_browser_')
            options.add_argument(f"--user-data-dir={temp_profile_dir}")
            
            # Create driver
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=options
            )
            
            # Navigate to URL
            driver.get(url)
            
            # Wait for page load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Perform risk assessment
            risk_level, page_analysis = self._assess_page_risk(driver, url)
            
            # Store session
            self.active_sessions[session_id] = {
                'driver': driver,
                'url': url,
                'created_at': datetime.now(),
                'incognito': incognito,
                'profile_dir': temp_profile_dir,
                'risk_level': risk_level
            }
            
            # Schedule auto-close if not incognito
            if not incognito:
                threading.Timer(
                    self.security_config['session_timeout_minutes'] * 60,
                    self.close_session,
                    args=(session_id,)
                ).start()
            
            logger.info(f"Created secure session {session_id} for {url} (incognito: {incognito})")
            
            return {
                'success': True,
                'session_id': session_id,
                'risk_level': risk_level,
                'page_analysis': page_analysis
            }
            
        except Exception as e:
            logger.error(f"Secure session creation failed: {e}")
            # Cleanup if partial creation
            if 'temp_profile_dir' in locals():
                shutil.rmtree(temp_profile_dir, ignore_errors=True)
            return {'success': False, 'error': str(e)}
    
    def _assess_page_risk(self, driver, url: str) -> Tuple[str, Dict]:
        """Assess page risk level"""
        try:
            # Check trusted domains
            domain = url.split('//')[-1].split('/')[0]
            if any(t in domain for t in self.security_config['trusted_domains']):
                return 'low', {'message': 'Trusted domain, no analysis needed.'}
            
            # Basic analysis
            page_title = driver.title
            page_source = driver.page_source
            
            # Simple heuristic checks
            suspicious_keywords = ['phishing', 'malware', 'fake', 'scam']
            suspicion_score = sum(1 for kw in suspicious_keywords if kw in page_source.lower())
            
            # Metadata analysis
            meta_tags = driver.find_elements(By.TAG_NAME, 'meta')
            meta_analysis = [tag.get_attribute('content') for tag in meta_tags if tag.get_attribute('content')]
            
            risk_level = 'high' if suspicion_score > 2 else 'medium' if suspicion_score > 0 else 'low'
            
            analysis = {
                'title': page_title,
                'meta_count': len(meta_tags),
                'suspicion_score': suspicion_score,
                'risk_factors': [kw for kw in suspicious_keywords if kw in page_source.lower()]
            }
            
            return risk_level, analysis
            
        except Exception as e:
            logger.warning(f"Page risk assessment failed: {e}")
            return 'unknown', {'error': str(e)}
    
    def close_session(self, session_id: str) -> bool:
        """Close specific browser session and cleanup"""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions.pop(session_id)
        
        try:
            if 'driver' in session:
                session['driver'].quit()
            
            if 'profile_dir' in session and os.path.exists(session['profile_dir']):
                shutil.rmtree(session['profile_dir'], ignore_errors=True)
            
            logger.info(f"Closed secure session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Session close failed: {e}")
            return False
    
    def close_all_sessions(self):
        """Close all active sessions"""
        for session_id in list(self.active_sessions.keys()):
            self.close_session(session_id)
        logger.info("All secure sessions closed")

class StreamMonitoringEngine:
    """
    Enhanced stream monitoring engine with multi-source support
    and real-time deepfake detection
    """
    def __init__(self, detector_model: AdvancedDeepFakeDetector, 
                 face_analyzer: EnhancedFaceAnalyzer,
                 analytics: RealTimeAnalyticsEngine):
        self.detector_model = detector_model
        self.face_analyzer = face_analyzer
        self.analytics = analytics
        self.active_streams = {}
        self.monitoring_threads = {}
        self.stop_events = {}
        self.frame_buffers = {}
        self.detection_history = {}
        self.recording_enabled = True
        self.alert_threshold = 0.7
        logger.info("StreamMonitoringEngine initialized")
    
    def add_stream_source(self, source_id: str, source_config: Dict):
        """Add new stream source"""
        if source_id in self.active_streams:
            logger.warning(f"Stream source {source_id} already exists")
            return False
        
        self.active_streams[source_id] = source_config
        self.stop_events[source_id] = threading.Event()
        self.frame_buffers[source_id] = deque(maxlen=30)  # 1 second buffer at 30fps
        self.detection_history[source_id] = deque(maxlen=300)  # 10 seconds history
        
        logger.info(f"Added stream source {source_id}: {source_config}")
        return True
    
    def start_monitoring(self, source_id: str):
        """Start monitoring specific stream"""
        if source_id not in self.active_streams:
            logger.error(f"Unknown stream source {source_id}")
            return False
        
        if source_id in self.monitoring_threads:
            logger.warning(f"Monitoring already active for {source_id}")
            return False
        
        thread = threading.Thread(
            target=self._monitor_stream,
            args=(source_id,),
            daemon=True
        )
        self.monitoring_threads[source_id] = thread
        thread.start()
        
        logger.info(f"Started monitoring stream {source_id}")
        return True
    
    def _monitor_stream(self, source_id: str):
        """Monitor stream in separate thread"""
        config = self.active_streams[source_id]
        stop_event = self.stop_events[source_id]
        
        try:
            source = int(config['source_path']) if config['source_type'] == 'webcam' else config['source_path']
            cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open stream {source_id}")
            
            frame_count = 0
            last_detection_time = time.time()
            
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Stream {source_id} frame read failed")
                    time.sleep(1)
                    continue
                
                # Add to buffer
                self.frame_buffers[source_id].append(frame)
                
                frame_count += 1
                if frame_count % 5 != 0:  # Analyze every 5th frame (~6 fps)
                    continue
                
                # Perform analysis
                current_time = time.time()
                if current_time - last_detection_time < 1.0:  # Rate limit to 1/sec
                    continue
                
                analysis = self._analyze_frame(frame)
                self.detection_history[source_id].append(analysis)
                
                if analysis['overall_confidence'] > self.alert_threshold:
                    self._trigger_alert(source_id, analysis)
                    if self.recording_enabled:
                        self._record_suspicious_clip(source_id, analysis)
                
                # Update analytics
                self.analytics.update_metrics(analysis)
                
                last_detection_time = current_time
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Stream monitoring failed for {source_id}: {e}")
        finally:
            if source_id in self.monitoring_threads:
                del self.monitoring_threads[source_id]
    
    def _analyze_frame(self, frame) -> Dict:
        """Analyze single frame from stream"""
        analysis_start = time.time()
        
        face_analysis = self.face_analyzer.comprehensive_face_analysis(frame)
        face_conf = face_analysis.get('overall_confidence', 0.0)
        
        # Prepare for NN
        frame_tensor = self._prepare_frame(frame)
        if frame_tensor is not None:
            with torch.no_grad():
                frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)  # Add seq dim
                device = next(self.detector_model.parameters()).device
                frame_tensor = frame_tensor.to(device)
                output, _ = self.detector_model(frame_tensor)
                probs = F.softmax(output, dim=1)
                nn_conf = float(probs[0, 1])
        else:
            nn_conf = 0.0
        
        overall_conf = max(face_conf, nn_conf)
        
        result = {
            'analysis_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'confidence': overall_conf,
            'is_deepfake': overall_conf > 0.5,
            'processing_time': time.time() - analysis_start,
            'model_version': '3.0',
            'face_analysis': face_analysis,
            'neural_network_result': {'confidence': nn_conf}
        }
        
        return result
    
    def _prepare_frame(self, frame):
        """Prepare frame for neural network input"""
        try:
            resized = cv2.resize(frame, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            return transform(rgb)
        except Exception as e:
            logger.warning(f"Frame preparation failed: {e}")
            return None
    
    def _trigger_alert(self, source_id: str, analysis: Dict):
        """Trigger alert for suspicious detection"""
        logger.warning(f"Deepfake alert in stream {source_id}: confidence {analysis['confidence']:.2%}")
        # Could add email/sms alert here
    
    def _record_suspicious_clip(self, source_id: str, analysis: Dict):
        """Record suspicious clip from buffer"""
        try:
            buffer = list(self.frame_buffers[source_id])
            if not buffer:
                return
            
            clip_id = analysis['analysis_id']
            output_path = f"suspicious_clips/{clip_id}.mp4"
            os.makedirs('suspicious_clips', exist_ok=True)
            
            height, width = buffer[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
            
            for frame in buffer:
                writer.write(frame)
            
            writer.release()
            
            logger.info(f"Recorded suspicious clip: {output_path}")
            
            # Store in database
            record_result = {
                'file_path': output_path,
                'media_type': 'video',
                **analysis
            }
            self.database.store_analysis_result(record_result)
            
        except Exception as e:
            logger.error(f"Clip recording failed: {e}")
    
    def stop_monitoring(self, source_id: Optional[str] = None):
        """Stop monitoring specific or all streams"""
        if source_id:
            if source_id in self.stop_events:
                self.stop_events[source_id].set()
                if source_id in self.monitoring_threads:
                    self.monitoring_threads[source_id].join(timeout=5.0)
                logger.info(f"Stopped monitoring stream {source_id}")
        else:
            for sid in list(self.stop_events.keys()):
                self.stop_events[sid].set()
                if sid in self.monitoring_threads:
                    self.monitoring_threads[sid].join(timeout=5.0)
            logger.info("Stopped all stream monitoring")

class AuthenticationManager:
    """
    Enhanced authentication manager with TOTP support,
    account locking, and session management
    """
    def __init__(self, database: Optional[DatabaseManager] = None):
        self.database = database
        self.active_sessions: Dict[str, Dict] = {}
        self.security_config = {
            'max_login_attempts': 5,
            'lockout_duration_minutes': 30,
            'session_timeout_hours': 24,
            'token_length': 32,
            'password_min_length': 12,
            'require_special_chars': True
        }
        logger.info("AuthenticationManager initialized")
    
    def create_user(self, username: str, password: str, role: str = 'user', totp_secret: Optional[str] = None):
        """Create new user with hashed password"""
        try:
            if not self.database:
                return False
            
            if not self._validate_password_strength(password):
                return False
            
            salt = secrets.token_hex(16)
            password_hash = self._hash_password(password, salt)
            
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO users (username, password_hash, salt, role, totp_secret)
                    VALUES (?, ?, ?, ?, ?)
                ''', (username, password_hash, salt, role, totp_secret))
                conn.commit()
            
            logger.info(f"Created user {username} with role {role}")
            return True
            
        except sqlite3.IntegrityError:
            logger.warning(f"Username {username} already exists")
            return False
        except Exception as e:
            logger.error(f"User creation failed: {e}")
            return False
    
    def authenticate_user(self, username: str, password: str, totp_code: Optional[str] = None, ip_address: Optional[str] = None):
        """Authenticate user with multi-factor support"""
        try:
            if not self.database:
                return {'success': False, 'error': 'No database connection'}
            
            if self._is_user_locked(username):
                return {'success': False, 'error': 'Account locked'}
            
            with sqlite3.connect(self.database.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM users WHERE username = ?
                ''', (username,))
                user = cursor.fetchone()
                
                if not user:
                    self._log_login_attempt(username, ip_address, False, 'Invalid username')
                    return {'success': False, 'error': 'Invalid credentials'}
                
                # Verify password
                hashed = self._hash_password(password, user['salt'])
                if hashed != user['password_hash']:
                    self._increment_failed_attempts(username)
                    self._log_login_attempt(username, ip_address, False, 'Invalid password')
                    return {'success': False, 'error': 'Invalid credentials'}
                
                # Check if TOTP is required
                if user['totp_secret']:
                    if not totp_code:
                        return {'success': False, 'error': 'TOTP required', 'requires_totp': True}
                    if not self._verify_totp(user['totp_secret'], totp_code):
                        self._increment_failed_attempts(username)
                        self._log_login_attempt(username, ip_address, False, 'Invalid TOTP')
                        return {'success': False, 'error': 'Invalid TOTP code'}
                
                # Authentication successful
                self._reset_failed_attempts(username)
                
                # Create session
                session_info = self._create_session(user, ip_address)
                
                # Update last login
                cursor.execute('''
                    UPDATE users SET last_login = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (user['id'],))
                
                conn.commit()
                
                self._log_login_attempt(username, ip_address, True)
                
                logger.info(f"User '{username}' authenticated successfully")
                
                return {
                    'success': True,
                    'user_id': user['id'],
                    'username': user['username'],
                    'role': user['role'],
                    'session_token': session_info['session_token'],
                    'session_expires': session_info['expires_at']
                }
                
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return {'success': False, 'error': 'Authentication system error'}
    
    def _validate_password_strength(self, password):
        """Validate password meets security requirements"""
        try:
            if len(password) < self.security_config['password_min_length']:
                return False
            
            if self.security_config['require_special_chars']:
                special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
                if not any(char in special_chars for char in password):
                    return False
                
                # Require at least one number and one letter
                if not any(char.isdigit() for char in password):
                    return False
                
                if not any(char.isalpha() for char in password):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Password validation failed: {e}")
            return False
    
    def _hash_password(self, password, salt):
        """Hash password with salt using SHA-256"""
        try:
            return hashlib.sha256((password + salt).encode()).hexdigest()
        except Exception as e:
            logger.error(f"Password hashing failed: {e}")
            return None
    
    def _is_user_locked(self, username):
        """Check if user account is locked"""
        try:
            if not self.database:
                return False
            
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT failed_attempts, locked_until FROM users 
                    WHERE username = ?
                ''', (username,))
                
                result = cursor.fetchone()
                
                if not result:
                    return False
                
                failed_attempts, locked_until = result
                
                # Check if account is locked
                if locked_until:
                    lock_time = datetime.fromisoformat(locked_until)
                    if datetime.now() < lock_time:
                        return True
                    else:
                        # Lock expired, reset
                        cursor.execute('''
                            UPDATE users SET failed_attempts = 0, locked_until = NULL
                            WHERE username = ?
                        ''', (username,))
                        conn.commit()
                
                return False
                
        except Exception as e:
            logger.error(f"User lock check failed: {e}")
            return False
    
    def _increment_failed_attempts(self, username):
        """Increment failed login attempts and lock account if necessary"""
        try:
            if not self.database:
                return
            
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE users SET failed_attempts = failed_attempts + 1
                    WHERE username = ?
                ''', (username,))
                
                # Check if account should be locked
                cursor.execute('''
                    SELECT failed_attempts FROM users WHERE username = ?
                ''', (username,))
                
                result = cursor.fetchone()
                
                if result and result[0] >= self.security_config['max_login_attempts']:
                    lock_until = datetime.now() + timedelta(minutes=self.security_config['lockout_duration_minutes'])
                    
                    cursor.execute('''
                        UPDATE users SET locked_until = ?
                        WHERE username = ?
                    ''', (lock_until.isoformat(), username))
                    
                    logger.warning(f"Account '{username}' locked due to failed login attempts")
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed attempts increment failed: {e}")
    
    def _reset_failed_attempts(self, username):
        """Reset failed login attempts counter"""
        try:
            if not self.database:
                return
            
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE users SET failed_attempts = 0, locked_until = NULL
                    WHERE username = ?
                ''', (username,))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed attempts reset failed: {e}")
    
    def _create_session(self, user, ip_address=None):
        """Create user session"""
        try:
            session_token = secrets.token_urlsafe(self.security_config['token_length'])
            expires_at = datetime.now() + timedelta(hours=self.security_config['session_timeout_hours'])
            
            session_info = {
                'user_id': user['id'],
                'username': user['username'],
                'role': user['role'],
                'session_token': session_token,
                'created_at': datetime.now(),
                'expires_at': expires_at,
                'ip_address': ip_address
            }
            
            # Store in memory
            self.active_sessions[session_token] = session_info
            
            # Store in database
            if self.database:
                with sqlite3.connect(self.database.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT INTO user_sessions 
                        (user_id, session_token, expires_at, ip_address)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        user['id'],
                        session_token,
                        expires_at.isoformat(),
                        ip_address
                    ))
                    
                    conn.commit()
            
            return session_info
            
        except Exception as e:
            logger.error(f"Session creation failed: {e}")
            return None
    
    def validate_session(self, session_token):
        """Validate and refresh session"""
        try:
            # Check in-memory sessions first
            if session_token in self.active_sessions:
                session = self.active_sessions[session_token]
                
                # Check if session is expired
                if datetime.now() > session['expires_at']:
                    del self.active_sessions[session_token]
                    return None
                
                # Update last activity
                session['last_activity'] = datetime.now()
                return session
            
            # Check database sessions
            if self.database:
                with sqlite3.connect(self.database.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        SELECT us.*, u.username, u.role
                        FROM user_sessions us
                        JOIN users u ON us.user_id = u.id
                        WHERE us.session_token = ? AND us.is_active = 1
                    ''', (session_token,))
                    result = cursor.fetchone()
                    
                    if not result:
                        return None
                    
                    session = dict(result)
                    expires_at = datetime.fromisoformat(session['expires_at'])
                    
                    if datetime.now() > expires_at:
                        cursor.execute('UPDATE user_sessions SET is_active = 0 WHERE id = ?', (session['id'],))
                        conn.commit()
                        return None
                    
                    # Update last activity
                    cursor.execute('UPDATE user_sessions SET last_activity = CURRENT_TIMESTAMP WHERE id = ?', (session['id'],))
                    conn.commit()
                    
                    # Cache in memory
                    self.active_sessions[session_token] = {
                        'user_id': session['user_id'],
                        'username': session['username'],
                        'role': session['role'],
                        'expires_at': expires_at,
                        'last_activity': datetime.now(),
                        'ip_address': session.get('ip_address')
                    }
                    
                    return self.active_sessions[session_token]
            
        except Exception as e:
            logger.error(f"Session validation failed: {e}")
            return None
    
    def logout(self, session_token):
        """Logout user and invalidate session"""
        try:
            if session_token in self.active_sessions:
                del self.active_sessions[session_token]
            
            if self.database:
                with sqlite3.connect(self.database.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('UPDATE user_sessions SET is_active = 0 WHERE session_token = ?', (session_token,))
                    conn.commit()
            
            logger.info(f"Session {session_token} logged out successfully")
            return True
            
        except Exception as e:
            logger.error(f"Logout failed: {e}")
            return False
    
    def change_password(self, username, old_password, new_password):
        """Change user password"""
        try:
            if not self.database:
                return False
            
            if not self._validate_password_strength(new_password):
                return False
            
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('SELECT password_hash, salt FROM users WHERE username = ?', (username,))
                user = cursor.fetchone()
                
                if not user:
                    return False
                
                old_hash = self._hash_password(old_password, user[1])
                if old_hash != user[0]:
                    return False
                
                new_salt = secrets.token_hex(16)
                new_hash = self._hash_password(new_password, new_salt)
                
                cursor.execute('UPDATE users SET password_hash = ?, salt = ? WHERE username = ?', (new_hash, new_salt, username))
                conn.commit()
            
            logger.info(f"Password changed for user {username}")
            return True
            
        except Exception as e:
            logger.error(f"Password change failed: {e}")
            return False
    
    def _verify_totp(self, secret, code):
        """Verify TOTP code"""
        # Note: Requires 'pyotp' library, assume installed or add to dependencies
        try:
            import pyotp
            totp = pyotp.TOTP(secret)
            return totp.verify(code)
        except ImportError:
            logger.error("pyotp not available for TOTP verification")
            return False
        except Exception as e:
            logger.error(f"TOTP verification failed: {e}")
            return False
    
    def _log_login_attempt(self, username, ip_address, success, failure_reason=None):
        """Log login attempt"""
        try:
            if not self.database:
                return
            
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO login_attempts (username, ip_address, success, failure_reason)
                    VALUES (?, ?, ?, ?)
                ''', (username, ip_address, 1 if success else 0, failure_reason))
                conn.commit()
        except Exception as e:
            logger.error(f"Login attempt logging failed: {e}")

class DeepFakePlatformGUI:
    def __init__(self):
        self.database = DatabaseManager()
        self.analytics = RealTimeAnalyticsEngine(database=self.database)
        self.face_analyzer = EnhancedFaceAnalyzer()
        self.detector_model = AdvancedDeepFakeDetector()
        if torch.cuda.is_available():
            self.detector_model.to('cuda')
        self.detector_model.eval()
        self.browser_manager = SecureBrowserManager(database=self.database)
        self.stream_engine = StreamMonitoringEngine(self.detector_model, self.face_analyzer, self.analytics)
        self.auth_manager = AuthenticationManager(database=self.database)

        self.root = tk.Tk()
        self.root.title("Enhanced DeepFake Detection Platform v3.0")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.logged_in = False
        self.current_user = None

        self._create_login_screen()

    def _on_closing(self):
        self.browser_manager.close_all_sessions()
        if self.logged_in and self.current_user and 'session_token' in self.current_user:
            self.auth_manager.logout(self.current_user['session_token'])
        self.root.destroy()

    def _create_login_screen(self):
        self.login_frame = ttk.Frame(self.root)
        self.login_frame.pack(fill='both', expand=True, padx=20, pady=20)

        ttk.Label(self.login_frame, text="Username:", font=('Arial', 12)).pack(pady=10)
        self.username_entry = ttk.Entry(self.login_frame, font=('Arial', 12))
        self.username_entry.pack(pady=10)

        ttk.Label(self.login_frame, text="Password:", font=('Arial', 12)).pack(pady=10)
        self.password_entry = ttk.Entry(self.login_frame, show="*", font=('Arial', 12))
        self.password_entry.pack(pady=10)

        self.totp_label = ttk.Label(self.login_frame, text="TOTP Code (if enabled):", font=('Arial', 12))
        self.totp_entry = ttk.Entry(self.login_frame, font=('Arial', 12))

        self.login_button = ttk.Button(self.login_frame, text="Login", command=self._handle_login)
        self.login_button.pack(pady=20)

    def _handle_login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        totp = self.totp_entry.get() if self.totp_entry.winfo_ismapped() else None

        auth_result = self.auth_manager.authenticate_user(username, password, totp)

        if auth_result['success']:
            self.current_user = auth_result
            self.logged_in = True
            self.login_frame.destroy()
            self._create_main_interface()
        else:
            messagebox.showerror("Login Failed", auth_result.get('error', 'Unknown error'))
            if auth_result.get('requires_totp'):
                self.totp_label.pack(pady=10)
                self.totp_entry.pack(pady=10)

    def _create_main_interface(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)

        self.detection_tab = ttk.Frame(self.notebook)
        self.browser_tab = ttk.Frame(self.notebook)
        self.stream_tab = ttk.Frame(self.notebook)
        self.analytics_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.detection_tab, text='Media Detection')
        self.notebook.add(self.browser_tab, text='Secure Browsing')
        self.notebook.add(self.stream_tab, text='Stream Monitoring')
        self.notebook.add(self.analytics_tab, text='Analytics Dashboard')
        self.notebook.add(self.settings_tab, text='Settings & Privacy')

        self._setup_detection_tab()
        self._setup_browser_tab()
        self._setup_stream_tab()
        self._setup_analytics_tab()
        self._setup_settings_tab()

    def _setup_detection_tab(self):
        ttk.Label(self.detection_tab, text="Select Media File for DeepFake Detection", font=('Arial', 14)).pack(pady=20)

        self.file_path_var = tk.StringVar()
        ttk.Entry(self.detection_tab, textvariable=self.file_path_var, width=50).pack(pady=10)
        ttk.Button(self.detection_tab, text="Browse", command=self._browse_file).pack(pady=10)
        ttk.Button(self.detection_tab, text="Analyze Media", command=self._analyze_media).pack(pady=20)

        self.result_text = tk.Text(self.detection_tab, height=15, width=80)
        self.result_text.pack(pady=20)

    def _browse_file(self):
        file = filedialog.askopenfilename(filetypes=[("Media Files", "*.mp4 *.avi *.mov *.jpg *.png")])
        if file:
            self.file_path_var.set(file)

    def _analyze_media(self):
        file_path = self.file_path_var.get()
        if not file_path:
            messagebox.showerror("Error", "Please select a file")
            return

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Analyzing...\n")

        try:
            # Check if known sample
            known_label = self.database.check_known_sample(file_path)
            if known_label:
                result_str = f"Known sample detected: {known_label.upper()}\n"
                self.result_text.insert(tk.END, result_str)
                return

            is_video = file_path.lower().endswith(('.mp4', '.avi', '.mov'))
            frames = []
            if is_video:
                cap = cv2.VideoCapture(file_path)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                cap.release()
                selected_frames = frames[::max(1, len(frames)//10)]  # Select up to 10 frames
            else:
                frame = cv2.imread(file_path)
                selected_frames = [frame]

            confidences = []
            for frame in selected_frames:
                if frame is None:
                    continue
                face_analysis = self.face_analyzer.comprehensive_face_analysis(frame)
                face_conf = face_analysis.get('overall_confidence', 0.0)

                frame_tensor = self._prepare_frame_for_nn(frame)
                nn_conf = 0.0
                if frame_tensor is not None:
                    with torch.no_grad():
                        frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)
                        device = next(self.detector_model.parameters()).device
                        frame_tensor = frame_tensor.to(device)
                        output, _ = self.detector_model(frame_tensor)
                        probs = F.softmax(output, dim=1)
                        nn_conf = float(probs[0, 1])

                overall_conf = max(face_conf, nn_conf) if face_analysis['faces_detected'] > 0 else nn_conf
                confidences.append(overall_conf)

            if not confidences:
                raise Exception("No frames analyzed")

            avg_conf = np.mean(confidences)
            is_deepfake = avg_conf > 0.5

            result_str = f"Analysis Complete\nAverage Confidence (Deepfake Probability): {avg_conf:.2%}\nIs DeepFake: {'Yes' if is_deepfake else 'No'}\n"
            self.result_text.insert(tk.END, result_str)

            # Store result
            analysis_result = {
                'file_path': file_path,
                'media_type': 'video' if is_video else 'image',
                'confidence': avg_conf,
                'is_deepfake': is_deepfake,
                'processing_time': 0.0,  # Placeholder
                'model_version': '3.0',
                'face_analysis': face_analysis if not is_video else {},
                'neural_network_result': {'confidence': nn_conf if not is_video else np.mean(confidences)}
            }
            self.database.store_analysis_result(analysis_result)
            self.analytics.update_metrics(analysis_result)

        except Exception as e:
            messagebox.showerror("Analysis Error", str(e))
            self.result_text.insert(tk.END, f"Error: {str(e)}\n")

    def _prepare_frame_for_nn(self, frame):
        try:
            resized = cv2.resize(frame, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            return transform(rgb)
        except:
            return None

    def _setup_browser_tab(self):
        ttk.Label(self.browser_tab, text="Secure Browsing", font=('Arial', 14)).pack(pady=20)

        ttk.Label(self.browser_tab, text="Enter URL:").pack(pady=10)
        self.url_var = tk.StringVar()
        ttk.Entry(self.browser_tab, textvariable=self.url_var, width=50).pack(pady=10)

        ttk.Button(self.browser_tab, text="Open in Secure Browser", command=self._open_secure_browser).pack(pady=20)

        self.browser_result = tk.Text(self.browser_tab, height=15, width=80)
        self.browser_result.pack(pady=20)

    def _open_secure_browser(self):
        url = self.url_var.get()
        if not url:
            messagebox.showerror("Error", "Please enter a URL")
            return

        self.browser_result.delete(1.0, tk.END)
        self.browser_result.insert(tk.END, "Analyzing URL for legitimacy...\n")

        try:
            # First, analyze the URL in headless mode to determine risk
            # We pass incognito=False here because we want to run analysis in the background
            session_result = self.browser_manager.create_secure_session(url, incognito=False) 
            
            if session_result.get('success', False):
                risk_level = session_result['risk_level']
                self.browser_result.insert(tk.END, f"Initial Risk Assessment: {risk_level.upper()}\n")
                self.browser_result.insert(tk.END, f"Page Analysis: {json.dumps(session_result['page_analysis'], indent=2)}\n")

                # Close the initial headless session immediately after analysis
                if 'session_id' in session_result:
                    self.browser_manager.close_session(session_result['session_id'])

                if risk_level == 'low':
                    self.browser_result.insert(tk.END, "Link deemed legitimate. Opening in incognito mode...\n")
                    # Open a new session in incognito mode, which will be visible
                    incognito_session_result = self.browser_manager.create_secure_session(url, incognito=True)
                    
                    if incognito_session_result.get('success', False):
                        messagebox.showinfo("Secure Browsing", f"Opening {url} in a new incognito browser window.")
                        self.browser_result.insert(tk.END, f"Incognito Session ID: {incognito_session_result['session_id']}\n")
                        self.browser_result.insert(tk.END, "Please close the incognito window manually when done.\n")
                        # Do NOT close this session automatically, as the user is interacting with it.
                        # The user will close the browser window themselves.
                    else:
                        messagebox.showerror("Secure Browsing Error", f"Failed to open incognito browser: {incognito_session_result.get('error')}")
                        self.browser_result.insert(tk.END, f"Error opening incognito: {incognito_session_result.get('error')}\n")
                else:
                    self.browser_result.insert(tk.END, "Link identified as potentially risky. Not opening in incognito mode.\n")
                    messagebox.showwarning("Secure Browsing Warning", f"The link is identified as {risk_level.upper()} risk. Not opening in incognito mode.")
            else:
                self.browser_result.insert(tk.END, f"Error during initial analysis: {session_result.get('error')}\n")

        except Exception as e:
            self.browser_result.insert(tk.END, f"An unexpected error occurred: {str(e)}\n")
            messagebox.showerror("Secure Browsing Error", f"An unexpected error occurred: {str(e)}")


    def _setup_stream_tab(self):
        ttk.Label(self.stream_tab, text="Stream Monitoring", font=('Arial', 14)).pack(pady=20)
        ttk.Label(self.stream_tab, text="Enter webcam index (e.g., 0) or RTSP URL for IP camera. This monitors live streams for deepfakes in real-time, alerting and recording suspicious content.", wraplength=600, font=('Arial', 10)).pack(pady=10)

        ttk.Label(self.stream_tab, text="Stream Source:").pack(pady=10)
        self.stream_source_var = tk.StringVar()
        ttk.Entry(self.stream_tab, textvariable=self.stream_source_var, width=50).pack(pady=10)

        ttk.Button(self.stream_tab, text="Start Monitoring", command=self._start_stream).pack(pady=10)
        ttk.Button(self.stream_tab, text="Stop Monitoring", command=self._stop_stream).pack(pady=10)

        self.stream_status = tk.Text(self.stream_tab, height=15, width=80)
        self.stream_status.pack(pady=20)

    def _start_stream(self):
        source = self.stream_source_var.get()
        if not source:
            messagebox.showerror("Error", "Please enter a stream source")
            return

        source_id = "main_stream"
        source_config = {
            'source_type': 'webcam' if source.isdigit() else 'rtsp',
            'source_path': source
        }

        self.stream_engine.add_stream_source(source_id, source_config)
        self.stream_engine.start_monitoring(source_id)

        self.stream_status.insert(tk.END, "Stream monitoring started\n")

    def _stop_stream(self):
        self.stream_engine.stop_monitoring()
        self.stream_status.insert(tk.END, "Stream monitoring stopped\n")

    def _setup_analytics_tab(self):
        ttk.Label(self.analytics_tab, text="Analytics Dashboard", font=('Arial', 14)).pack(pady=20)

        ttk.Button(self.analytics_tab, text="Generate Report & Charts", command=self._generate_analytics_report).pack(pady=20)
        self.analytics_canvas_frame = ttk.Frame(self.analytics_tab)
        self.analytics_canvas_frame.pack(fill='both', expand=True)
    def _generate_analytics_report(self):
        report = self.analytics.generate_analytics_report()
        # Clear previous charts
        for widget in self.analytics_canvas_frame.winfo_children():
            widget.destroy()
        if not report:
            messagebox.showerror("Error", "No analytics data available")
            return
        # Create figures for charts
        fig1 = Figure(figsize=(5, 4))
        ax1 = fig1.add_subplot(111)
        ax1.bar(['Detections', 'Total Analyses'], [report['detection_statistics']['detection_count'], report['detection_statistics']['total_analyses']])
        ax1.set_title('Detection Statistics')
        canvas1 = FigureCanvasTkAgg(fig1, master=self.analytics_canvas_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(side=tk.LEFT, padx=10)
        fig2 = Figure(figsize=(5, 4))
        ax2 = fig2.add_subplot(111)
        media_types = list(report['detection_statistics']['media_types'].keys())
        counts = list(report['detection_statistics']['media_types'].values())
        ax2.pie(counts, labels=media_types, autopct='%1.1f%%')
        ax2.set_title('Media Types')
        canvas2 = FigureCanvasTkAgg(fig2, master=self.analytics_canvas_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(side=tk.LEFT, padx=10)
        if 'detection_trend' in report and report['detection_trend']:
            fig3 = Figure(figsize=(5, 4))
            ax3 = fig3.add_subplot(111)
            times = [t['timestamp'] for t in report['detection_trend']]
            confs = [t['confidence'] for t in report['detection_trend']]
            ax3.plot(times, confs)
            ax3.set_title('Detection Trend')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Confidence')
            canvas3 = FigureCanvasTkAgg(fig3, master=self.analytics_canvas_frame)
            canvas3.draw()
            canvas3.get_tk_widget().pack(side=tk.LEFT, padx=10)
    def _setup_settings_tab(self):
        ttk.Label(self.settings_tab, text="Settings & Privacy", font=('Arial', 14)).pack(pady=20)
        ttk.Button(self.settings_tab, text="Change Password", command=self._change_password).pack(pady=10)
        ttk.Button(self.settings_tab, text="Cleanup Old Data", command=self._cleanup_data).pack(pady=10)
        ttk.Button(self.settings_tab, text="Logout", command=self._logout).pack(pady=10)
    def _change_password(self):
        if not self.current_user:
            return
        old_pass = simpledialog.askstring("Change Password", "Old Password:", show="*")
        new_pass = simpledialog.askstring("Change Password", "New Password:", show="*")
        if new_pass:
            success = self.auth_manager.change_password(self.current_user['username'], old_pass, new_pass)
            if success:
                messagebox.showinfo("Success", "Password changed")
            else:
                messagebox.showerror("Error", "Password change failed")
    def _cleanup_data(self):
        if messagebox.askyesno("Confirm", "Cleanup old data?"):
            self.database.cleanup_old_data()
            messagebox.showinfo("Success", "Old data cleaned")
    def _logout(self):
        if self.current_user and 'session_token' in self.current_user:
            self.auth_manager.logout(self.current_user['session_token'])
        self.root.destroy()
if __name__ == "__main__":
    print("Starting Enhanced DeepFake Detection Platform v3.0")
    platform = DeepFakePlatformGUI()
    platform.root.mainloop()
