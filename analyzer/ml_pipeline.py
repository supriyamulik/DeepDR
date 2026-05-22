import os
import sys
import torch
import numpy as np
import cv2
import importlib.util
from django.conf import settings
from tensorflow.keras.models import load_model

def load_seg_model(module_name, device):
    module_path = os.path.join(settings.BASE_DIR, module_name)
    checkpoint_path = os.path.join(module_path, "files", "checkpoint.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found for {module_name}: {checkpoint_path}")
        return None
    
    # Dynamic import of model architecture
    spec = importlib.util.spec_from_file_location("module_model", os.path.join(module_path, "model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    model = mod.build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    model.eval()
    return model

class ModelManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelManager, cls).__new__(cls, *args, **kwargs)
            cls._instance.initialized = False
        return cls._instance

    def initialize(self):
        if self.initialized:
            return
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[DeepDR] Initializing models on device: {self.device}")
        
        # Load Classification Model
        clf_model_path = os.path.join(settings.BASE_DIR, 'Classification', 'model', 'model.h5')
        if os.path.exists(clf_model_path):
            print(f"[DeepDR] Loading Keras classification model from {clf_model_path}...")
            self.clf_model = load_model(clf_model_path, compile=False)
        else:
            print(f"[DeepDR] Error: Keras model not found at {clf_model_path}")
            self.clf_model = None
            
        # Load 6 Segmentation Models
        self.seg_modules = {
            "blood_vessel": "Blood Vessel Segmentation",
            "hemorrhage": "Haemorage Segmentation",
            "hard_exudate": "Hard Exudate Segmentation",
            "microaneurysm": "Microanuerism Segmentation",
            "optic_disc": "Optical Disc Segmentation",
            "soft_exudate": "Soft Exudate Segmentation"
        }
        
        self.seg_models = {}
        for key, folder_name in self.seg_modules.items():
            print(f"[DeepDR] Loading segmentation model for {folder_name}...")
            model = load_seg_model(folder_name, self.device)
            if model:
                self.seg_models[key] = model
                
        self.initialized = True
        print("[DeepDR] ModelManager initialized successfully.")

    def predict_classification(self, img):
        if self.clf_model is None:
            return "No Diabetic Retinopathy", 0, [100.0, 0.0, 0.0, 0.0, 0.0]
            
        img_clf = cv2.resize(img, (64, 64))
        img_clf = np.reshape(img_clf, [1, 64, 64, 3])
        
        prediction = self.clf_model.predict(img_clf, verbose=0)
        r = prediction[0][0]
        
        class_idx = int(np.round(r)) - 2
        class_idx = max(0, min(4, class_idx))
        
        DR_LEVELS = {
            0: 'No Diabetic Retinopathy',
            1: 'Mild Diabetic Retinopathy',
            2: 'Moderate Diabetic Retinopathy',
            3: 'Severe Diabetic Retinopathy',
            4: 'Proliferative Diabetic Retinopathy'
        }
        diagnosis = DR_LEVELS[class_idx]
        
        # Feature 3: Calculate continuous probability breakdown using a normal distribution
        # Classes: 0 (r=2.0), 1 (r=3.0), 2 (r=4.0), 3 (r=5.0), 4 (r=6.0)
        sigma = 0.5
        probs = []
        for idx in [2, 3, 4, 5, 6]:
            val = np.exp(-0.5 * ((idx - r) / sigma)**2)
            probs.append(float(val))
        
        total = sum(probs)
        if total > 0:
            probs = [round((p / total) * 100, 1) for p in probs]
        else:
            probs = [0.0] * 5
            probs[class_idx] = 100.0
            
        return diagnosis, class_idx, probs

    def predict_segmentation(self, img_path):
        # Read image
        img_seg_ori = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_seg_ori is None:
            raise ValueError(f"Could not load image at {img_path}")
            
        # Resize to 512x512 for standard U-Net input
        img_seg = cv2.resize(img_seg_ori, (512, 512))
        x = np.transpose(img_seg, (2, 0, 1))
        x = x / 255.0
        x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        
        masks = {}
        for key, model in self.seg_models.items():
            with torch.no_grad():
                pred_y = model(x)
                pred_y = torch.sigmoid(pred_y)
                pred_y = pred_y[0][0].cpu().numpy()
                pred_y = (pred_y > 0.5).astype(np.uint8) * 255
                masks[key] = pred_y
                
        return img_seg_ori, masks

    def run_full_pipeline(self, original_path, output_filename):
        """
        Runs both classification and segmentation, creates overlays, and returns results.
        """
        # Ensure directories exist
        overlays_dir = os.path.join(settings.MEDIA_ROOT, 'scans', 'overlays')
        os.makedirs(overlays_dir, exist_ok=True)
        
        # Load image for classification
        img = cv2.imread(original_path)
        if img is None:
            raise ValueError(f"Could not read image for classification: {original_path}")
            
        diagnosis, class_idx, probs = self.predict_classification(img)
        
        # Run segmentation
        img_seg_ori, masks = self.predict_segmentation(original_path)
        
        # Resize original image to 512x512 for visualization overlays
        img_512 = cv2.resize(img_seg_ori, (512, 512))
        
        # Color definitions (BGR)
        COLORS = {
            "blood_vessel": (0, 255, 0),        # Green
            "hemorrhage": (0, 127, 255),        # Orange
            "hard_exudate": (0, 255, 255),      # Yellow
            "microaneurysm": (0, 0, 255),       # Red
            "optic_disc": (255, 0, 255),        # Purple
            "soft_exudate": (255, 191, 0),      # Light Blue
        }
        
        # 1. Create transparent overlays for each mask type
        mask_urls = {}
        for key, mask in masks.items():
            # Create a 4-channel BGRA image initialized to completely transparent
            overlay_bgra = np.zeros((512, 512, 4), dtype=np.uint8)
            color = COLORS[key]
            
            # Where mask is positive, set color and opacity (255 = fully opaque, controlled in CSS)
            idx = (mask > 127)
            overlay_bgra[idx, 0] = color[0]
            overlay_bgra[idx, 1] = color[1]
            overlay_bgra[idx, 2] = color[2]
            overlay_bgra[idx, 3] = 255
            
            mask_filename = f"mask_{key}_{output_filename}.png"
            mask_path = os.path.join(overlays_dir, mask_filename)
            cv2.imwrite(mask_path, overlay_bgra)
            mask_urls[key] = f"scans/overlays/{mask_filename}"
            
        # 2. Create combined overlay
        combined_img = img_512.copy()
        for key, mask in masks.items():
            color = COLORS[key]
            idx = (mask > 127)
            # Blend color onto combined image with alpha 0.4
            combined_img[idx] = (combined_img[idx] * 0.6 + np.array(color) * 0.4).astype(np.uint8)
            
        combined_filename = f"combined_{output_filename}"
        combined_path = os.path.join(overlays_dir, combined_filename)
        cv2.imwrite(combined_path, combined_img)
        combined_url = f"scans/overlays/{combined_filename}"
        
        return {
            "diagnosis": diagnosis,
            "severity_index": class_idx,
            "confidence_breakdown": {
                "probs": probs,
                "score": float(np.round(class_idx + 2.0, 3)) # Reconstruct approximate raw score
            },
            "masks": mask_urls,
            "combined_overlay": combined_url
        }

    def run_explainability(self, original_path, output_filename, seg_mask_urls):
        """
        Generate Grad-CAM heatmaps and segmentation cross-validations dynamically.
        """
        from analyzer.gradcam import (
            generate_multilayer_gradcam,
            generate_gradcam_for_layer,
            generate_segmentation_overlays
        )
        
        # Ensure directories exist
        overlays_dir = os.path.join(settings.MEDIA_ROOT, 'scans', 'overlays')
        os.makedirs(overlays_dir, exist_ok=True)
        
        # Load image
        img = cv2.imread(original_path)
        if img is None:
            raise ValueError(f"Could not read image for Grad-CAM: {original_path}")
            
        img_clf = cv2.resize(img, (64, 64))
        img_clf = np.reshape(img_clf, [1, 64, 64, 3])
        
        # Generate multi-layer Grad-CAM
        comparison = None
        try:
            comparison = generate_multilayer_gradcam(
                self.clf_model, img_clf, original_path,
                overlays_dir, output_filename,
                threshold=0.3
            )
        except Exception as e:
            print(f"[Grad-CAM WARNING] Multi-layer comparison failed: {e}")
            import traceback
            traceback.print_exc()
            
        # Convert absolute overlay paths to relative URLs based on how Django serves MEDIA
        # Note: generate_multilayer_gradcam returns paths like 'results/filename.png',
        # we will adjust them to 'scans/overlays/filename.png' for the media URL.
        if comparison:
            for layer_data in comparison['layers']:
                for method_name, method_data in layer_data['methods'].items():
                    if method_data.get('overlay_url'):
                        method_data['overlay_url'] = method_data['overlay_url'].replace('results/', 'scans/overlays/')
                    if method_data.get('heatmap_url'):
                        method_data['heatmap_url'] = method_data['heatmap_url'].replace('results/', 'scans/overlays/')
                        
        # Generate Segmentation Overlays
        gradcam_seg_overlays = {}
        try:
            if comparison:
                best_layer = comparison.get('best_layer')
                best_method = comparison.get('best_method', 'pre_activation')
                if best_layer:
                    best_heatmap = generate_gradcam_for_layer(
                        self.clf_model, img_clf, best_layer,
                        gradient_method=best_method, threshold=0.3
                    )
                    
                    if best_heatmap is not None:
                        # seg_mask_urls gives relative media URLs, we need absolute paths
                        seg_paths = {}
                        for module_name, url in seg_mask_urls.items():
                            if url:
                                file_name = os.path.basename(url)
                                seg_paths[module_name] = os.path.join(overlays_dir, file_name)
                            
                        seg_overlay_results = generate_segmentation_overlays(
                            best_heatmap, seg_paths,
                            overlays_dir, output_filename,
                            original_image_path=original_path
                        )
                        
                        for module_name, rel_url in seg_overlay_results.items():
                            gradcam_seg_overlays[module_name] = rel_url.replace('results/', 'scans/overlays/')
        except Exception as e:
            print(f"[Grad-CAM WARNING] Segmentation cross-validation failed: {e}")
            
        return {
            "gradcam_comparison": comparison,
            "gradcam_seg_overlays": gradcam_seg_overlays
        }
