"""
Grad-CAM Explainability Module for DeepDR Classification Model.

Generates Gradient-weighted Class Activation Maps (Grad-CAM) to visualize
which retinal regions influenced the DR severity prediction.

Supports both:
  - Single-output regression models (Dense(1))
  - Multi-class softmax classification models (Dense(N, activation='softmax'))

Enhanced with:
  - Multi-layer Grad-CAM experimentation (compare across Conv2D layers)
  - Three regression-aware gradient methods (direct, squared, pre-activation)
  - Heatmap quality improvements (thresholding, cubic interpolation, percentile norm)
  - Shortcut learning diagnostics (border/corner activation analysis)
  - Segmentation mask cross-validation overlays

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017.
"""

import numpy as np
import cv2
import tensorflow as tf
import os


# =============================================================================
# Layer Discovery
# =============================================================================

def get_last_conv_layer(model):
    """
    Dynamically identify the last Conv2D layer in a Keras model.

    Searches backwards through model.layers to find the final convolutional
    layer, which contains the richest spatial feature maps for Grad-CAM.

    Args:
        model: A compiled or uncompiled tf.keras.Model instance.

    Returns:
        The name (str) of the last Conv2D layer found.

    Raises:
        ValueError: If no Conv2D layer exists in the model.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError(
        "No Conv2D layer found in model. Grad-CAM requires at least one "
        "convolutional layer to generate activation maps."
    )


def get_all_conv_layers(model):
    """
    Identify ALL Conv2D layers in the model, ordered from first to last.

    Also includes MaxPooling2D layers as intermediate comparison points,
    since they represent spatially-reduced feature maps that may produce
    different Grad-CAM attention patterns.

    Args:
        model: A tf.keras.Model instance.

    Returns:
        List of dicts with keys: 'name', 'type', 'output_shape', 'label'.
        Ordered from first (shallowest) to last (deepest).
    """
    layers = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            layers.append({
                'name': layer.name,
                'type': 'Conv2D',
                'output_shape': str(layer.output.shape),
                'label': f"{layer.name} ({layer.output.shape[1]}×{layer.output.shape[2]}×{layer.output.shape[3]})"
            })
        elif isinstance(layer, tf.keras.layers.MaxPooling2D):
            layers.append({
                'name': layer.name,
                'type': 'MaxPooling2D',
                'output_shape': str(layer.output.shape),
                'label': f"{layer.name} ({layer.output.shape[1]}×{layer.output.shape[2]}×{layer.output.shape[3]})"
            })
    return layers


# =============================================================================
# Core Grad-CAM Generation (Enhanced)
# =============================================================================

def generate_gradcam_for_layer(model, img_array, layer_name,
                                gradient_method='pre_activation',
                                threshold=0.3):
    """
    Generate a Grad-CAM heatmap for a specific layer with configurable
    gradient method and activation thresholding.

    Args:
        model: A tf.keras.Model instance.
        img_array: Preprocessed image as numpy array of shape (1, H, W, C).
        layer_name: Name of the target layer (Conv2D or MaxPooling2D).
        gradient_method: One of:
            - 'direct': loss = predictions[:, 0]  (standard regression)
            - 'squared': loss = tf.square(predictions[:, 0])  (amplified signal)
            - 'pre_activation': gradients w.r.t. output before final activation
              (bypasses ReLU gradient clipping — recommended for this model)
        threshold: Float 0.0–1.0. Activations below this are zeroed to
                   suppress weak noise. Default 0.3.

    Returns:
        heatmap: A normalized 2D numpy array (values 0.0–1.0), or None on error.
    """
    try:
        target_layer = model.get_layer(layer_name)

        # --- Build sub-models ---
        # Feature extractor: input → target layer output
        feature_extractor = tf.keras.Model(model.inputs, target_layer.output)

        # Classifier: target layer output → model output
        # We build this by chaining layers after the target layer
        conv_input = tf.keras.Input(shape=target_layer.output.shape[1:])
        x = conv_input
        start_idx = model.layers.index(target_layer) + 1

        # For pre_activation method, we need to identify the layer
        # just before the final activation to tap the pre-activation signal
        pre_activation_layer_name = None
        if gradient_method == 'pre_activation':
            # Find the last Dense layer before the final Activation
            for i in range(len(model.layers) - 1, -1, -1):
                if isinstance(model.layers[i], tf.keras.layers.Dense):
                    pre_activation_layer_name = model.layers[i].name
                    break

        # Build the classifier sub-model
        layer_outputs = {}
        for layer in model.layers[start_idx:]:
            x = layer(x)
            layer_outputs[layer.name] = x

        # Choose outputs based on gradient method
        if gradient_method == 'pre_activation' and pre_activation_layer_name and pre_activation_layer_name in layer_outputs:
            # Output both: pre-activation (for gradient) and final (for info)
            classifier = tf.keras.Model(conv_input, [layer_outputs[pre_activation_layer_name], x])
        else:
            classifier = tf.keras.Model(conv_input, x)

        # --- Forward pass with gradient recording ---
        img_tensor = tf.cast(img_array, tf.float32)
        with tf.GradientTape() as tape:
            conv_outputs = feature_extractor(img_tensor)
            tape.watch(conv_outputs)

            if gradient_method == 'pre_activation' and pre_activation_layer_name and pre_activation_layer_name in layer_outputs:
                pre_act_output, predictions = classifier(conv_outputs)
                # Use pre-activation output as gradient target
                # This bypasses the final ReLU that clips gradient signal
                loss = pre_act_output[:, 0]
            else:
                predictions = classifier(conv_outputs)
                # Adapt to model output shape
                if predictions.shape[-1] == 1:
                    if gradient_method == 'squared':
                        loss = tf.square(predictions[:, 0])
                    else:  # 'direct'
                        loss = predictions[:, 0]
                else:
                    # Multi-class softmax: select the predicted class
                    pred_index = tf.argmax(predictions[0])
                    loss = predictions[:, pred_index]

        # --- Compute gradients ---
        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            print(f"[Grad-CAM WARNING] Gradients are None for layer '{layer_name}' "
                  f"with method '{gradient_method}'. Graph may be disconnected.")
            return None

        # --- Global Average Pooling of gradients → importance weights ---
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # --- Weight each feature map by its importance ---
        conv_out = conv_outputs[0]  # Remove batch dimension
        heatmap = conv_out @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # --- ReLU: keep only positive influence features ---
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap.numpy()

        # --- Activation thresholding (suppress weak noise) ---
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
            heatmap[heatmap < threshold] = 0

        # --- Re-normalize after thresholding ---
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val

        # --- Percentile-based contrast enhancement ---
        # Stretch the dynamic range so that the top activations pop
        if np.max(heatmap) > 0:
            p_low = np.percentile(heatmap[heatmap > 0], 5) if np.any(heatmap > 0) else 0
            p_high = np.percentile(heatmap[heatmap > 0], 95) if np.any(heatmap > 0) else 1
            if p_high > p_low:
                heatmap = np.clip((heatmap - p_low) / (p_high - p_low), 0, 1)

        return heatmap

    except Exception as e:
        print(f"[Grad-CAM ERROR] Failed for layer '{layer_name}', "
              f"method '{gradient_method}': {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# Backward-Compatible API (preserves existing workflow)
# =============================================================================

def generate_gradcam_heatmap(model, img_array):
    """
    Generate a Grad-CAM heatmap for a given input image.
    Backward-compatible wrapper — uses pre_activation method by default.

    Args:
        model: A tf.keras.Model instance (loaded with load_model).
        img_array: Preprocessed image as a numpy array of shape (1, H, W, C).

    Returns:
        heatmap: A normalized 2D numpy array (values 0.0–1.0), or None on error.
    """
    try:
        last_conv = get_last_conv_layer(model)
        return generate_gradcam_for_layer(
            model, img_array, last_conv,
            gradient_method='pre_activation',
            threshold=0.3
        )
    except Exception as e:
        print(f"[Grad-CAM ERROR] Failed to generate heatmap: {e}")
        return None


# =============================================================================
# Heatmap Visualization (Enhanced Quality)
# =============================================================================

def overlay_heatmap(image_path, heatmap, save_path, alpha=0.4):
    """
    Overlay a Grad-CAM heatmap onto the original retinal fundus image.

    Resizes the heatmap to match the original image dimensions, applies
    the JET colormap, and blends the two images.

    Args:
        image_path: Path to the original retinal image file.
        heatmap: A 2D numpy array (values 0.0–1.0) from generate_gradcam_heatmap.
        save_path: File path where the blended overlay image will be saved.
        alpha: Heatmap blending weight (default 0.4). The original image
               weight is (1 - alpha), i.e., 0.6.

    Returns:
        True if the overlay was saved successfully, False otherwise.
    """
    try:
        # Read original image at full resolution
        original = cv2.imread(image_path)
        if original is None:
            print(f"[Grad-CAM ERROR] Could not read image: {image_path}")
            return False

        h, w = original.shape[:2]

        # Use INTER_CUBIC for sharper, less blurry heatmap resize
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
        heatmap_resized = np.clip(heatmap_resized, 0, 1)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Apply JET colormap for visualization
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Blend: original (60%) + heatmap (40%)
        overlay = cv2.addWeighted(original, 1.0 - alpha, heatmap_colored, alpha, 0)

        cv2.imwrite(save_path, overlay)
        return True

    except Exception as e:
        print(f"[Grad-CAM ERROR] Failed to create overlay: {e}")
        return False


def save_raw_heatmap(heatmap, save_path):
    """
    Save a standalone colored heatmap image for research and publication use.

    Produces a clean JET-colored heatmap without the original image underneath,
    suitable for thesis screenshots, research paper figures, and presentations.

    Args:
        heatmap: A 2D numpy array (values 0.0–1.0) from generate_gradcam_heatmap.
        save_path: File path where the raw heatmap image will be saved.

    Returns:
        True if the raw heatmap was saved successfully, False otherwise.
    """
    try:
        # Scale to standard research output size with INTER_CUBIC
        heatmap_resized = cv2.resize(heatmap, (512, 512), interpolation=cv2.INTER_CUBIC)
        heatmap_resized = np.clip(heatmap_resized, 0, 1)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Apply JET colormap
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        cv2.imwrite(save_path, heatmap_colored)
        return True

    except Exception as e:
        print(f"[Grad-CAM ERROR] Failed to save raw heatmap: {e}")
        return False


# =============================================================================
# Task 5: Shortcut Learning Diagnostics
# =============================================================================

def diagnose_shortcut_learning(heatmap, image_path=None, border_fraction=0.10):
    """
    Analyze a Grad-CAM heatmap for signs of shortcut learning.

    Checks whether activation is concentrated on:
    - Image borders (within border_fraction of edges)
    - Black corners (common in circular fundus images)
    - Optic disc only (single bright region)
    - Spatially uniform (no localized attention)

    Args:
        heatmap: 2D numpy array (0.0–1.0) from Grad-CAM.
        image_path: Optional path to the original image for corner analysis.
        border_fraction: Fraction of image dimensions to consider as "border".

    Returns:
        Dict with diagnostic metrics:
        - border_activation_pct: % of total activation near edges
        - corner_activation_pct: % of total activation in corners
        - spatial_entropy: Entropy of activation distribution (low = concentrated)
        - peak_count: Number of distinct activation peaks
        - potential_shortcut: Boolean flag
        - warnings: List of warning strings
    """
    diagnostics = {
        'border_activation_pct': 0.0,
        'corner_activation_pct': 0.0,
        'spatial_entropy': 0.0,
        'peak_count': 0,
        'potential_shortcut': False,
        'warnings': []
    }

    if heatmap is None or np.max(heatmap) == 0:
        diagnostics['warnings'].append("Empty or null heatmap")
        diagnostics['potential_shortcut'] = True
        return diagnostics

    h, w = heatmap.shape
    total_activation = np.sum(heatmap)

    if total_activation == 0:
        diagnostics['potential_shortcut'] = True
        diagnostics['warnings'].append("Zero total activation")
        return diagnostics

    # --- Border activation analysis ---
    border_h = max(1, int(h * border_fraction))
    border_w = max(1, int(w * border_fraction))

    # Create border mask
    border_mask = np.zeros_like(heatmap, dtype=bool)
    border_mask[:border_h, :] = True      # Top
    border_mask[-border_h:, :] = True     # Bottom
    border_mask[:, :border_w] = True      # Left
    border_mask[:, -border_w:] = True     # Right

    border_activation = np.sum(heatmap[border_mask])
    border_pct = (border_activation / total_activation) * 100
    diagnostics['border_activation_pct'] = float(round(border_pct, 1))

    if border_pct > 50:
        diagnostics['warnings'].append(
            f"High border activation: {border_pct:.1f}% of attention is within "
            f"{border_fraction*100:.0f}% of image edges"
        )

    # --- Corner activation analysis ---
    corner_mask = np.zeros_like(heatmap, dtype=bool)
    corner_size_h = max(1, int(h * 0.15))
    corner_size_w = max(1, int(w * 0.15))
    corner_mask[:corner_size_h, :corner_size_w] = True       # Top-left
    corner_mask[:corner_size_h, -corner_size_w:] = True      # Top-right
    corner_mask[-corner_size_h:, :corner_size_w] = True      # Bottom-left
    corner_mask[-corner_size_h:, -corner_size_w:] = True     # Bottom-right

    corner_activation = np.sum(heatmap[corner_mask])
    corner_pct = (corner_activation / total_activation) * 100
    diagnostics['corner_activation_pct'] = float(round(corner_pct, 1))

    if corner_pct > 30:
        diagnostics['warnings'].append(
            f"High corner activation: {corner_pct:.1f}% — possible black corner shortcut"
        )

    # --- Spatial entropy (concentration measure) ---
    flat = heatmap.flatten()
    flat_norm = flat / np.sum(flat)
    flat_norm = flat_norm[flat_norm > 0]  # Remove zeros for log
    entropy = -np.sum(flat_norm * np.log2(flat_norm + 1e-10))
    max_entropy = np.log2(h * w)  # Maximum possible entropy
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    diagnostics['spatial_entropy'] = float(round(normalized_entropy, 3))

    if normalized_entropy > 0.95:
        diagnostics['warnings'].append(
            "Very high spatial entropy — activation is nearly uniform (no localized attention)"
        )

    # --- Peak count (distinct activation regions) ---
    binary = (heatmap > 0.5).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(binary)
    peak_count = num_labels - 1  # Subtract background
    diagnostics['peak_count'] = int(peak_count)

    if peak_count == 0:
        diagnostics['warnings'].append("No strong activation peaks detected (all below 0.5)")

    # --- Overall shortcut flag ---
    shortcut_signals = 0
    if border_pct > 40:
        shortcut_signals += 1
    if corner_pct > 20:
        shortcut_signals += 1
    if normalized_entropy > 0.95:
        shortcut_signals += 1
    if peak_count == 0:
        shortcut_signals += 1

    diagnostics['potential_shortcut'] = shortcut_signals >= 2

    if diagnostics['potential_shortcut']:
        print(f"[SHORTCUT DIAGNOSTIC] ⚠ Potential shortcut learning detected!")
        for w in diagnostics['warnings']:
            print(f"  → {w}")
        print(f"  Border: {border_pct:.1f}%, Corners: {corner_pct:.1f}%, "
              f"Entropy: {normalized_entropy:.3f}, Peaks: {peak_count}")

    return diagnostics


# =============================================================================
# Task 6: Segmentation Cross-Validation
# =============================================================================

def overlay_gradcam_on_segmentation(heatmap, seg_mask_path, save_path,
                                      original_image_path=None):
    """
    Create a composite image overlaying Grad-CAM onto a segmentation mask.

    Produces a side-by-side or blended visualization showing whether the
    CNN's attention overlaps with clinically relevant lesion regions.

    Args:
        heatmap: 2D numpy array (0.0–1.0) from Grad-CAM.
        seg_mask_path: Path to the binary segmentation mask image.
        save_path: Where to save the composite image.
        original_image_path: Optional original image for 3-panel composite.

    Returns:
        True if saved successfully, False otherwise.
    """
    try:
        seg_mask = cv2.imread(seg_mask_path, cv2.IMREAD_GRAYSCALE)
        if seg_mask is None:
            print(f"[Grad-CAM ERROR] Could not read segmentation mask: {seg_mask_path}")
            return False

        h, w = seg_mask.shape

        # Resize heatmap to match segmentation mask dimensions
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
        heatmap_resized = np.clip(heatmap_resized, 0, 1)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Create colored heatmap
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Create colored segmentation mask (green for lesion regions)
        seg_colored = np.zeros((h, w, 3), dtype=np.uint8)
        seg_colored[seg_mask > 127] = [0, 255, 0]  # Green for lesion pixels

        # Create overlap visualization:
        # - Green = segmentation only (lesion present, no attention)
        # - Red/Yellow = Grad-CAM only (attention but no lesion)
        # - White = overlap (attention matches lesion)
        overlap = np.zeros((h, w, 3), dtype=np.uint8)

        seg_binary = seg_mask > 127
        heatmap_binary = heatmap_resized > 0.3

        # Green channel: segmentation mask
        overlap[:, :, 1] = np.where(seg_binary, 255, 0)
        # Red channel: heatmap attention
        overlap[:, :, 2] = np.where(heatmap_binary, heatmap_uint8, 0)
        # Blue channel: overlap regions
        overlap_mask = seg_binary & heatmap_binary
        overlap[:, :, 0] = np.where(overlap_mask, 180, 0)

        # If original image available, create 3-panel composite
        if original_image_path:
            original = cv2.imread(original_image_path)
            if original is not None:
                original_resized = cv2.resize(original, (w, h))
                # Blend heatmap onto segmentation
                seg_with_heatmap = cv2.addWeighted(
                    seg_colored, 0.5, heatmap_colored, 0.5, 0
                )
                # Stack horizontally: seg mask | overlap | heatmap on seg
                composite = np.hstack([seg_colored, overlap, seg_with_heatmap])
                cv2.imwrite(save_path, composite)
                return True

        # Fallback: just the overlap panel
        cv2.imwrite(save_path, overlap)
        return True

    except Exception as e:
        print(f"[Grad-CAM ERROR] Segmentation overlay failed: {e}")
        return False


# =============================================================================
# Multi-Layer Pipeline (Task 1 orchestration)
# =============================================================================

# Gradient methods to test
GRADIENT_METHODS = ['direct', 'squared', 'pre_activation']

# Labels for gradient methods in UI
GRADIENT_METHOD_LABELS = {
    'direct': 'Method A: Direct (loss = output)',
    'squared': 'Method B: Squared (loss = output²)',
    'pre_activation': 'Method C: Pre-Activation (bypass final ReLU)'
}


def generate_multilayer_gradcam(model, img_array, image_path,
                                  results_folder, filename,
                                  threshold=0.3):
    """
    Generate Grad-CAM outputs for ALL candidate layers × ALL gradient methods.

    Saves comparison images and returns structured results for the UI.

    Args:
        model: tf.keras.Model instance.
        img_array: Preprocessed image (1, H, W, C).
        image_path: Path to original full-resolution image.
        results_folder: Directory to save output images.
        filename: Original image filename for naming outputs.
        threshold: Activation threshold for noise suppression.

    Returns:
        Dict with structure:
        {
            'layers': [
                {
                    'name': 'conv2d_3',
                    'label': 'conv2d_3 (29×29×64)',
                    'type': 'Conv2D',
                    'methods': {
                        'direct': {'overlay_url': ..., 'heatmap_url': ..., 'diagnostics': ...},
                        'squared': {...},
                        'pre_activation': {...}
                    }
                },
                ...
            ],
            'best_method': 'pre_activation',
            'best_layer': 'conv2d_3'
        }
    """
    all_layers = get_all_conv_layers(model)
    base_name = os.path.splitext(filename)[0]

    results = {
        'layers': [],
        'best_method': 'pre_activation',
        'best_layer': all_layers[-1]['name'] if all_layers else None
    }

    # Position labels for file naming
    position_labels = []
    conv_count = sum(1 for l in all_layers if l['type'] == 'Conv2D')
    conv_idx = 0
    for layer_info in all_layers:
        if layer_info['type'] == 'Conv2D':
            if conv_idx == conv_count - 1:
                position_labels.append('last')
            elif conv_idx == conv_count - 2:
                position_labels.append('second_last')
            elif conv_idx == conv_count - 3:
                position_labels.append('third_last')
            else:
                position_labels.append(f'conv_{conv_idx}')
            conv_idx += 1
        else:
            position_labels.append(layer_info['name'].replace(' ', '_'))

    best_score = -1

    for layer_idx, layer_info in enumerate(all_layers):
        layer_name = layer_info['name']
        pos_label = position_labels[layer_idx]

        layer_result = {
            'name': layer_name,
            'label': layer_info['label'],
            'type': layer_info['type'],
            'position': pos_label,
            'methods': {}
        }

        for method in GRADIENT_METHODS:
            print(f"[Grad-CAM] Generating: layer={layer_name}, method={method}")

            heatmap = generate_gradcam_for_layer(
                model, img_array, layer_name,
                gradient_method=method,
                threshold=threshold
            )

            method_result = {
                'overlay_url': None,
                'heatmap_url': None,
                'diagnostics': None,
                'method_label': GRADIENT_METHOD_LABELS[method]
            }

            if heatmap is not None:
                # Save overlay
                overlay_name = f"gradcam_{pos_label}_{method}_{filename}"
                overlay_path = os.path.join(results_folder, overlay_name)
                overlay_heatmap(image_path, heatmap, overlay_path)
                method_result['overlay_url'] = f'results/{overlay_name}'

                # Save raw heatmap
                heatmap_name = f"heatmap_only_{pos_label}_{method}_{filename}"
                heatmap_path = os.path.join(results_folder, heatmap_name)
                save_raw_heatmap(heatmap, heatmap_path)
                method_result['heatmap_url'] = f'results/{heatmap_name}'

                # Run shortcut diagnostics
                diag = diagnose_shortcut_learning(heatmap, image_path)
                method_result['diagnostics'] = diag

                # Score this combination (lower border %, higher peaks = better)
                score = (100 - diag['border_activation_pct']) + (diag['peak_count'] * 10)
                if score > best_score:
                    best_score = score
                    results['best_method'] = method
                    results['best_layer'] = layer_name

            layer_result['methods'][method] = method_result

        results['layers'].append(layer_result)

    # Also save the standard named outputs for Task 1 compatibility
    _save_standard_outputs(model, img_array, image_path, results_folder,
                           filename, all_layers, position_labels, threshold)

    print(f"[Grad-CAM] Best combination: layer={results['best_layer']}, "
          f"method={results['best_method']}")

    return results


def _save_standard_outputs(model, img_array, image_path, results_folder,
                           filename, all_layers, position_labels, threshold):
    """
    Save the standard-named outputs requested in the task spec:
      gradcam_last.png, gradcam_second_last.png, gradcam_third_last.png
      heatmap_only_last.png, etc.

    Uses the best gradient method (pre_activation) for these standard outputs.
    """
    for layer_idx, layer_info in enumerate(all_layers):
        pos = position_labels[layer_idx]
        if pos not in ('last', 'second_last', 'third_last'):
            continue

        heatmap = generate_gradcam_for_layer(
            model, img_array, layer_info['name'],
            gradient_method='pre_activation',
            threshold=threshold
        )

        if heatmap is not None:
            # Standard overlay
            overlay_path = os.path.join(results_folder, f"gradcam_{pos}.png")
            overlay_heatmap(image_path, heatmap, overlay_path)

            # Standard raw heatmap
            heatmap_path = os.path.join(results_folder, f"heatmap_only_{pos}.png")
            save_raw_heatmap(heatmap, heatmap_path)


def generate_segmentation_overlays(heatmap, seg_results_paths, results_folder,
                                     filename, original_image_path=None):
    """
    Generate Grad-CAM × Segmentation overlay comparisons for all available masks.

    Args:
        heatmap: 2D numpy array from Grad-CAM (best method).
        seg_results_paths: Dict of {module_name: mask_file_path}.
        results_folder: Where to save overlay composites.
        filename: Original image filename for naming.
        original_image_path: Path to original image for 3-panel composite.

    Returns:
        Dict of {module_name: overlay_relative_url}.
    """
    overlays = {}

    if heatmap is None:
        return overlays

    for module_name, mask_path in seg_results_paths.items():
        try:
            overlay_name = f"gradcam_seg_{module_name.replace(' ', '_')}_{filename}"
            overlay_path = os.path.join(results_folder, overlay_name)

            success = overlay_gradcam_on_segmentation(
                heatmap, mask_path, overlay_path,
                original_image_path=original_image_path
            )

            if success:
                overlays[module_name] = f'results/{overlay_name}'

        except Exception as e:
            print(f"[Grad-CAM ERROR] Seg overlay for {module_name}: {e}")

    return overlays
