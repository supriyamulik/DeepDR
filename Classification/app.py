from flask import Flask, render_template, request, jsonify, url_for
import os
import sys
import torch
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import importlib.util
from dotenv import load_dotenv
import groq

from gradcam import (
    generate_gradcam_heatmap,
    overlay_heatmap,
    save_raw_heatmap,
    generate_multilayer_gradcam,
    generate_segmentation_overlays,
    generate_gradcam_for_layer,
    get_last_conv_layer
)

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'upload')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'static', 'results')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

def generate_clinical_summary(diagnosis, certainty, gradcam_data):
    load_dotenv(os.path.join(PROJECT_ROOT, '.env'), override=True)
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key or api_key.strip() == "" or api_key == "your_api_key_here":
        return "⚠️ **Groq API Key not configured.** Please add your API key to the `.env` file to enable AI clinical summaries."
    
    try:
        groq_client = groq.Groq(api_key=api_key)
    except Exception as e:
        return f"⚠️ Failed to initialize Groq client: {e}"
    
    diagnostics = "No anomaly detected."

    prompt = f"""You are an expert ophthalmologist AI. Explain this AI diagnosis in simple, patient-friendly language.
Diagnosis: {diagnosis}
AI Confidence: {certainty}%

Provide a 3-bullet-point explanation focusing on:
1. What this diagnosis generally means.
2. What the AI's confidence level tells us.
3. A brief disclaimer that this is an AI assistant, not a doctor.
Keep it extremely concise, empathetic, and professional. Format with Markdown bullets."""

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"⚠️ Failed to generate summary: {e}"

# Classification Model
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.h5')
clf_model = load_model(MODEL_PATH, compile=False)

DR_LEVELS = {
    0: 'No Diabetic Retinopathy',
    1: 'Mild Diabetic Retinopathy',
    2: 'Moderate Diabetic Retinopathy',
    3: 'Severe Diabetic Retinopathy',
    4: 'Proliferative Diabetic Retinopathy'
}

SEG_MODULES = [
    "Blood Vessel Segmentation",
    "Haemorage Segmentation",
    "Hard Exudate Segmentation",
    "Microanuerism Segmentation",
    "Optical Disc Segmentation",
    "Soft Exudate Segmentation"
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_seg_model(module_name):
    module_path = os.path.join(PROJECT_ROOT, module_name)
    checkpoint_path = os.path.join(module_path, "files", "checkpoint.pth")
    
    if not os.path.exists(checkpoint_path):
        return None
    
    # Dynamic import to avoid naming conflicts
    spec = importlib.util.spec_from_file_location("module_model", os.path.join(module_path, "model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    model = mod.build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    model.eval()
    return model

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    image_file = request.files['file']
    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save original image
    filename = image_file.filename
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(original_path)
    
    # 1. Classification
    img = cv2.imread(original_path)
    img_clf = cv2.resize(img, (64, 64))
    img_clf = np.reshape(img_clf, [1, 64, 64, 3])
    
    prediction = clf_model.predict(img_clf, verbose=0)
    r = prediction[0][0]
    class_idx = int(np.round(r)) - 2
    class_idx = max(0, min(4, class_idx))
    diagnosis = DR_LEVELS[class_idx]

    # --- Prediction Certainty ---
    # For regression output: certainty based on proximity to nearest class center
    # For softmax output: certainty = max probability * 100
    output_shape = clf_model.output_shape
    if output_shape[-1] == 1:
        distance = abs(float(r) - round(float(r)))
        certainty = round(max(0.0, 100.0 * (1.0 - distance)), 1)
    else:
        certainty = round(float(np.max(prediction)) * 100.0, 1)

    # --- Grad-CAM Explainability (backward-compatible) ---
    gradcam_overlay_url = None
    gradcam_raw_url = None
    try:
        heatmap = generate_gradcam_heatmap(clf_model, img_clf)
        if heatmap is not None:
            overlay_name = f"gradcam_overlay_{filename}"
            overlay_path = os.path.join(app.config['RESULTS_FOLDER'], overlay_name)
            overlay_heatmap(original_path, heatmap, overlay_path)
            gradcam_overlay_url = url_for('static', filename=f'results/{overlay_name}')

            raw_name = f"gradcam_raw_{filename}"
            raw_path = os.path.join(app.config['RESULTS_FOLDER'], raw_name)
            save_raw_heatmap(heatmap, raw_path)
            gradcam_raw_url = url_for('static', filename=f'results/{raw_name}')
    except Exception as e:
        print(f"[Grad-CAM WARNING] Could not generate explainability map: {e}")

    # --- Multi-Layer Grad-CAM Comparison (Research Mode) ---
    gradcam_comparison = None
    try:
        comparison = generate_multilayer_gradcam(
            clf_model, img_clf, original_path,
            app.config['RESULTS_FOLDER'], filename,
            threshold=0.3
        )
        # Convert file paths to URLs
        if comparison:
            for layer_data in comparison['layers']:
                for method_name, method_data in layer_data['methods'].items():
                    if method_data['overlay_url']:
                        method_data['overlay_url'] = url_for(
                            'static', filename=method_data['overlay_url']
                        )
                    if method_data['heatmap_url']:
                        method_data['heatmap_url'] = url_for(
                            'static', filename=method_data['heatmap_url']
                        )
            gradcam_comparison = comparison
    except Exception as e:
        print(f"[Grad-CAM WARNING] Multi-layer comparison failed: {e}")
        import traceback
        traceback.print_exc()

    # 2. Segmentation
    img_seg_ori = cv2.imread(original_path, cv2.IMREAD_COLOR)
    img_seg = cv2.resize(img_seg_ori, (512, 512))
    x = np.transpose(img_seg, (2, 0, 1))
    x = x/255.0
    x = np.expand_dims(x, axis=0)
    x = torch.from_numpy(x.astype(np.float32)).to(device)
    
    seg_results = {}
    seg_mask_paths = {}  # Track actual file paths for Grad-CAM overlay
    for module_name in SEG_MODULES:
        model = load_seg_model(module_name)
        if model:
            with torch.no_grad():
                pred_y = model(x)
                pred_y = torch.sigmoid(pred_y)
                pred_y = pred_y[0][0].cpu().numpy()
                pred_y = (pred_y > 0.5).astype(np.uint8) * 255
            
            mask_name = f"mask_{module_name.replace(' ', '_')}_{filename}"
            mask_path = os.path.join(app.config['RESULTS_FOLDER'], mask_name)
            cv2.imwrite(mask_path, pred_y)
            short_name = module_name.split(" ")[0]
            seg_results[short_name] = url_for('static', filename=f'results/{mask_name}')
            seg_mask_paths[short_name] = mask_path

    # --- Grad-CAM × Segmentation Cross-Validation (Task 6) ---
    gradcam_seg_overlays = {}
    try:
        if heatmap is not None and seg_mask_paths:
            # Use the best heatmap (pre_activation on last conv layer)
            best_heatmap = heatmap
            if gradcam_comparison:
                best_layer = gradcam_comparison.get('best_layer')
                best_method = gradcam_comparison.get('best_method', 'pre_activation')
                if best_layer:
                    best_heatmap_candidate = generate_gradcam_for_layer(
                        clf_model, img_clf, best_layer,
                        gradient_method=best_method, threshold=0.3
                    )
                    if best_heatmap_candidate is not None:
                        best_heatmap = best_heatmap_candidate

            seg_overlay_results = generate_segmentation_overlays(
                best_heatmap, seg_mask_paths,
                app.config['RESULTS_FOLDER'], filename,
                original_image_path=original_path
            )
            for module_name, rel_url in seg_overlay_results.items():
                gradcam_seg_overlays[module_name] = url_for(
                    'static', filename=rel_url
                )
    except Exception as e:
        print(f"[Grad-CAM WARNING] Segmentation cross-validation failed: {e}")
        
    clinical_summary = generate_clinical_summary(diagnosis, certainty, gradcam_comparison)
        
    return jsonify({
        "diagnosis": diagnosis,
        "prediction_certainty": certainty,
        "original_image": url_for('static', filename=f'upload/{filename}'),
        "gradcam_overlay": gradcam_overlay_url,
        "gradcam_raw": gradcam_raw_url,
        "gradcam_comparison": gradcam_comparison,
        "gradcam_seg_overlays": gradcam_seg_overlays,
        "clinical_summary": clinical_summary,
        "masks": seg_results
    })

if __name__ == '__main__':
    app.run(debug=True, port=8080)
