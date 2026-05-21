from flask import Flask, render_template, request, jsonify, url_for
import os
import sys
import torch
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import importlib.util

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'upload')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'static', 'results')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

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

    # 2. Segmentation
    img_seg_ori = cv2.imread(original_path, cv2.IMREAD_COLOR)
    img_seg = cv2.resize(img_seg_ori, (512, 512))
    x = np.transpose(img_seg, (2, 0, 1))
    x = x/255.0
    x = np.expand_dims(x, axis=0)
    x = torch.from_numpy(x.astype(np.float32)).to(device)
    
    seg_results = {}
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
            seg_results[module_name.split(" ")[0]] = url_for('static', filename=f'results/{mask_name}')
        
    return jsonify({
        "diagnosis": diagnosis,
        "original_image": url_for('static', filename=f'upload/{filename}'),
        "masks": seg_results
    })

if __name__ == '__main__':
    app.run(debug=True, port=8080)
