import cv2
import numpy as np
import torch
from transformers import AutoModelForDepthEstimation, AutoImageProcessor
import os

# Ensure the 'outputs' directory exists
os.makedirs('outputs-de', exist_ok=True)

# Load Depth Anything model and processor from Hugging Face
def load_depth_anything_model():
    depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    depth_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    return depth_model, depth_processor

# Function to generate a depth map using Depth Anything from Hugging Face
def generate_depth_map_with_depth_anything(image, model, processor):
    # Convert image to RGB and process it using Hugging Face processor
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image_rgb, return_tensors="pt")

    # Predict depth map with the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Debug: Print the output structure to see what it contains
    print(f"Model output keys: {outputs.keys()}")  # This will show all keys in the output object
    
    # Access the correct attribute for depth map
    depth_map_tensor = outputs.predicted_depth.squeeze(0)  # Try using 'predicted_depth' instead of 'depths'
    
    # Convert depth tensor to numpy array
    depth_map = depth_map_tensor.cpu().numpy()

    # Normalize depth map to [0, 255] for visualization
    depth_map = np.uint8(255 * (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)))
    
    return depth_map

# NPE Algorithm - Naturalness Preserved Enhancement
def npe_algorithm(image):
    blur = cv2.GaussianBlur(image, (7, 7), 0)
    illumination = blur
    illumination_3d = illumination
    reflectance = image / (illumination_3d + 1e-5)
    illumination_3d = np.log(illumination_3d + 1)
    reflectance = np.log(reflectance + 1)
    enhanced_image = np.exp(illumination_3d + reflectance)
    enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)
    return enhanced_image

# LIME Algorithm - Low-Light Image Enhancement
def lime_algorithm(image):
    # Step 1: Compute the initial illumination map
    illumination = np.max(image, axis=2).astype(np.float32) / 255.0  # Normalize to [0, 1]

    # Step 2: Enhance the illumination map using a structural constraint
    illumination = cv2.GaussianBlur(illumination, (15, 15), 0)  # Smooth the illumination map
    refined_illumination = np.clip(illumination, 0.1, 1)  # Avoid overly dark regions

    # Step 3: Adjust the RGB image based on the refined illumination
    adjusted_image = np.zeros_like(image, dtype=np.float32)
    for c in range(3):  # Process each channel independently
        adjusted_image[:, :, c] = image[:, :, c] / (refined_illumination + 1e-5)

    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)

    # Step 4: Noise reduction using BM3D on the Y channel
    ycbcr = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycbcr)

    # Simulate BM3D denoising (use a substitute like OpenCV's denoising)
    y_denoised = cv2.fastNlMeansDenoising(y, None, 10, 7, 21)

    # Merge the denoised Y channel with Cr and Cb channels
    denoised_ycbcr = cv2.merge([y_denoised, cr, cb])
    enhanced_image = cv2.cvtColor(denoised_ycbcr, cv2.COLOR_YCrCb2BGR)

    return enhanced_image

# FBE Algorithm - Fusion-Based Enhancement
def fbe_algorithm(image):
    # Step 1: Convert to grayscale and estimate illumination using morphological closing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    kernel = np.ones((15, 15), np.uint8)  # Larger kernel for smoother illumination
    illumination = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Increase illumination slightly by adding a small offset
    illumination += 0.1  # Increase illumination slightly for brighter results
    illumination = np.clip(illumination, 0.1, 1.0)  # Clip to avoid excessive brightness

    # Step 2: Decompose image into reflectance and illumination components
    reflectance = image.astype(np.float32) / (illumination[:, :, None] + 1e-5)

    # Normalize reflectance to avoid excessive values
    reflectance = cv2.normalize(reflectance, None, 0, 1, cv2.NORM_MINMAX)

    # Step 3: Enhance the illumination using adaptive histogram equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    illumination_enhanced = clahe.apply((illumination * 255).astype(np.uint8))
    illumination_enhanced = illumination_enhanced.astype(np.float32) / 255.0

    # Step 4: Apply a sigmoid function for further enhancement
    sigmoid_illumination = 1 / (1 + np.exp(-10 * (illumination_enhanced - 0.5)))

    # Step 5: Fuse the enhanced illumination with the reflectance
    fused_image = reflectance * sigmoid_illumination[:, :, None]

    # Normalize the fused image to prevent artifacts
    fused_image = np.clip(fused_image, 0, 1)

    # Step 6: Convert to uint8 for final output
    enhanced_image = (fused_image * 255.0).astype(np.uint8)
    return enhanced_image

# BIMEF Algorithm - Bio-Inspired Multi-Exposure Fusion
def bimef_algorithm(image):
    image_blur = cv2.GaussianBlur(image, (9, 9), 0)
    enhanced_image = cv2.addWeighted(image, 0.7, image_blur, 0.3, 0)
    enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)
    return enhanced_image

# RRM Algorithm - Robust Retinex Model
def rrm_algorithm(image):
    illumination = cv2.GaussianBlur(image, (9, 9), 0)
    # print('illumination:', illumination.shape)
    illumination_3d = cv2.merge([illumination[:, :, 0], illumination[:, :, 1], illumination[:, :, 2]])
    # print('illumination_3d:', illumination_3d.shape)
    reflectance = image / (illumination_3d + 1e-5)
    # print('ref:', reflectance.shape)
    enhanced_image = reflectance * illumination_3d
    # print('en_img:', enhanced_image.shape)
    enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)
    return enhanced_image

# SD Algorithm - Sequential Decomposition
def sd_algorithm(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    illumination = cv2.GaussianBlur(gray, (9, 9), 0)
    illumination_3d = cv2.merge([illumination] * 3)
    reflectance = image / (illumination_3d + 1e-5)
    enhanced_image = reflectance * illumination_3d
    enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)
    return enhanced_image

def fof_algorithm(image):
    # Ensure the input image is float32 for processing
    image = image.astype(np.float32) / 255.0

    # Step 1: Illumination estimation using Gaussian blur
    illumination = cv2.GaussianBlur(image, (15, 15), 0)

    # Step 2: Reflectance estimation (simple ratio)
    reflectance = image / (illumination + 1e-5)

    # Step 3: Clip reflectance to avoid oversaturation
    reflectance = np.clip(reflectance, 0, 1)

    # Step 4: Enhance illumination using a weighted average
    enhanced_illumination = cv2.addWeighted(image, 0.6, illumination, 0.4, 0)

    # Step 5: Dynamic fusion of reflectance and enhanced illumination
    enhanced_image = reflectance * enhanced_illumination

    # Step 6: Normalize and convert back to uint8
    enhanced_image = np.clip(enhanced_image * 255.0, 0, 255).astype(np.uint8)

    return enhanced_image


# IB Algorithm - Illumination Boost
def ib_algorithm(image):
    illumination = np.log(image + 1)
    enhanced_image = np.exp(illumination)
    enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)
    return enhanced_image
 
# AIE Algorithm - Adaptive Image Enhancement
def aie_algorithm(image):
    # Step 1: Convert RGB to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32) / 255.0  # Normalize V to [0, 1]

    # Step 2: Estimate illumination using guided filter (preserve edges)
    illumination = cv2.ximgproc.guidedFilter(guide=v, src=v, radius=15, eps=0.01)

    # Step 3: Adaptive enhancement of the V component
    enhanced_v1 = v / (illumination + 1e-5)  # Adaptive contrast enhancement
    enhanced_v2 = np.power(v, 0.8)  # Dark region enhancement

    # Normalize both enhanced versions to [0, 1]
    enhanced_v1 = cv2.normalize(enhanced_v1, None, 0, 1, cv2.NORM_MINMAX)
    enhanced_v2 = cv2.normalize(enhanced_v2, None, 0, 1, cv2.NORM_MINMAX)

    # Step 4: Fuse the two enhanced versions
    fused_v = 0.5 * enhanced_v1 + 0.5 * enhanced_v2

    # Step 5: Apply edge-preserving smoothing to remove spots
    fused_v = cv2.ximgproc.guidedFilter(guide=fused_v, src=fused_v, radius=5, eps=0.005)

    # Step 6: Replace the V component with the fused version
    hsv[:, :, 2] = np.clip(fused_v * 255, 0, 255).astype(np.uint8)

    # Step 7: Convert back to RGB color space
    enhanced_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return enhanced_image

# CR Algorithm - Camera Response Model
def cr_algorithm(image):
    enhanced_image = np.clip(image * 1.5, 0, 255).astype(np.uint8)
    return enhanced_image

# SDD Algorithm - Semi-Decoupled Decomposition
def sdd_algorithm(image):
    illumination = cv2.GaussianBlur(image, (9, 9), 0)
    illumination_3d = cv2.merge([illumination[:, :, 0], illumination[:, :, 1], illumination[:, :, 2]])
    reflectance = image / (illumination_3d + 1e-5)
    enhanced_image = reflectance * illumination_3d
    enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)
    return enhanced_image

# RBMP Algorithm - Retinex-Based Multiphase
def rbmp_algorithm(image):
    # Ensure the input image is float32 for processing
    image = image.astype(np.float32) / 255.0

    # Step 1: Illumination estimation using Gaussian blur
    illumination = cv2.GaussianBlur(image, (15, 15), 0)  # Larger kernel for smoother illumination

    # Step 2: Compute logarithms of the original image and illumination
    log_image = np.log1p(image)  # Use log1p for numerical stability (log(1 + x))
    log_illumination = np.log1p(illumination)

    # Step 3: Modified LIP subtraction
    lip_result = log_image - log_illumination
    lip_result = cv2.normalize(lip_result, None, 0, 1, cv2.NORM_MINMAX)  # Normalize to [0, 1]

    # Step 4: Gamma-corrected sigmoid function
    gamma = 2.2  # Gamma value for contrast control
    sigmoid_result = 1 / (1 + np.exp(-gamma * (lip_result - 0.5)))  # Sigmoid centered at 0.5

    # Step 5: Normalize the sigmoid result to [0, 255]
    sigmoid_result = cv2.normalize(sigmoid_result, None, 0, 1, cv2.NORM_MINMAX)

    # Step 6: Combine sigmoid result with original image for final enhancement
    enhanced_image = cv2.addWeighted(image, 0.5, sigmoid_result, 0.5, 0)  # Weighted blending
    enhanced_image = np.clip(enhanced_image * 255, 0, 255).astype(np.uint8)  # Convert to uint8

    return enhanced_image

# Function to save the results in the 'outputs' folder
def save_results(original_image, algorithms, model, processor):
    for name, func in algorithms.items():
        print(f"Processing algorithm: {name}...")
        try:
            enhanced_image = func(original_image)
            # Generate depth map using Depth Anything
            depth_map = generate_depth_map_with_depth_anything(enhanced_image, model, processor)
            # Save enhanced image and depth map
            cv2.imwrite(f'outputs-de/{name}_enhanced.jpg', enhanced_image)
            cv2.imwrite(f'outputs-de/Depth_map_{name}.jpg', depth_map)
        except Exception as e:
            print(f"Error processing {name}: {e}")

# Main function to run all algorithms and save the outputs
def main(image_path):
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Unable to load the image from path: {image_path}")
        return
    
    # Load Depth Anything model and processor
    model, processor = load_depth_anything_model()
    
    # Algorithms for image enhancement
    algorithms = {
        "NPE": npe_algorithm,
        "LIME": lime_algorithm,
        "FBE": fbe_algorithm,
        "BIMEF": bimef_algorithm,
        "RRM": rrm_algorithm,
        "SD": sd_algorithm,
        "FOF": fof_algorithm,
        "IB": ib_algorithm,
        "AIE": aie_algorithm,
        "CR": cr_algorithm,
        "SDD": sdd_algorithm,
        "RBMP": rbmp_algorithm,
    }
    
    # Process and save results
    save_results(original_image, algorithms, model, processor)
    print("All enhanced images and depth maps have been saved in the 'outputs' folder.")

if __name__ == "__main__":
    main('/data2/dse411a/project3/team2/sample_images/day_empty.jpg')  # Update this path
