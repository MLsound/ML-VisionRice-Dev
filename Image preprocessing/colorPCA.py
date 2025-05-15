# python colorPCA.py -i /path/to/your/input_image.jpg -o /path/to/save/result.png
# For testing: 
# '../../data/samples/3,5m90/209_101_24.JPG'
# '../../data/samples/3,5m90/503_118_29.JPG' 
# '../../data/samples/3,5m90/503_119_43.JPG'
# python colorPCA.py -i '../../data/samples/3,5m90/209_101_24.JPG'
# python colorPCA.py -i '../../data/raw/CLAHE/2_209_205_50_JPG.rf.dcf6a84732d63af4945585a50533cd01.jpg'
# python colorPCA.py -i '../../data/raw/CLAHE/2_503_118_29_JPG.rf.356a3084e7bcdb80995e397f446758c2.jpg'

import cv2
import numpy as np
from sklearn.decomposition import PCA
import argparse
import os

def scale_channel(channel):
    """Scales a channel to 0-255 range for visualization."""
    # Perform min-max scaling
    return cv2.normalize(channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype('uint8')

def process_image(input_path, output_path):
    """
    Reads an image, calculates ExG, performs PCA on RGB,
    and saves a new image with R=PC1, G=ExG, B=PC2.

    Args:
        input_path (str): Path to the input image file.
        output_path (str): Path to save the processed image file.

    Returns:
        bool: True if processing was successful, False otherwise.
    """
    print(f"Reading image: {input_path}")
    # Read the image using OpenCV (reads in BGR order)
    img_bgr = cv2.imread(input_path)

    if img_bgr is None:
        print(f"Error: Could not read image file {input_path}")
        return False

    if len(img_bgr.shape) != 3 or img_bgr.shape[2] != 3:
        print(f"Error: Input image must be a 3-channel (RGB/BGR) image.")
        return False

    h, w, _ = img_bgr.shape
    print(f"Image dimensions: {w}x{h}")

    # Convert to float32 for calculations
    img_bgr_float = img_bgr.astype(np.float32)

    # Split channels (remember OpenCV is BGR)
    b, g, r = cv2.split(img_bgr_float)

    # --- 1. Calculate Excess Green (ExG) ---
    print("Calculating Excess Green (ExG)...")
    # ExG = 2G - R - B
    exg = 2.0 * g - r - b
    # Scale ExG to 0-255 for visualization
    exg_scaled = scale_channel(exg)
    print("ExG calculation complete.")

    # --- 2. Perform PCA on original RGB ---
    print("Performing PCA on RGB channels...")
    # Reshape the image for PCA: (height * width, 3_channels)
    # We need RGB order for conventional understanding, although PCA is invariant to column order
    # Let's stack R, G, B for clarity before reshaping
    img_rgb_float = cv2.cvtColor(img_bgr_float, cv2.COLOR_BGR2RGB)
    rgb_reshaped = img_rgb_float.reshape(-1, 3)

    # Apply PCA to find the first 2 principal components
    pca = PCA(n_components=2)
    pca.fit(rgb_reshaped)
    pca_result = pca.transform(rgb_reshaped) # Shape: (h*w, 2)

    # Extract and reshape PC1 and PC2 back to image dimensions
    pc1_flat = pca_result[:, 0]
    pc2_flat = pca_result[:, 1]
    pc1 = pc1_flat.reshape(h, w)
    pc2 = pc2_flat.reshape(h, w)

    # Scale PC1 and PC2 to 0-255 for visualization
    pc1_scaled = scale_channel(pc1)
    pc2_scaled = scale_channel(pc2)

    # Save PCA channels as separate images for inspection
    pca1_path = os.path.splitext(output_path)[0] + "_pca1.png"
    pca2_path = os.path.splitext(output_path)[0] + "_pca2.png"
    exg_path = os.path.splitext(output_path)[0] + "_exg.png"
    cv2.imwrite(pca1_path, pc1_scaled)
    cv2.imwrite(pca2_path, pc2_scaled)
    cv2.imwrite(exg_path, exg_scaled)
    print(f"PCA complete. Explained variance ratio: {pca.explained_variance_ratio_}")

    # --- 3. Combine channels into the output image ---
    # Target: R channel = PC1, G channel = ExG, B channel = PC2
    #print("Combining channels: R=PC1, G=ExG, B=PC2")
    print("Combining channels: R=ExG, G=PC2, B=PC1")
    output_image_bgr = cv2.merge((pc1_scaled, pc2_scaled, exg_scaled))
    # OpenCV's merge expects (Blue, Green, Red) order


    # --- 4. Save the output image ---
    print(f"Saving processed image to: {output_path}")
    try:
        cv2.imwrite(output_path, output_image_bgr)
        print("Image saved successfully.")
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False

# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image: Calculate ExG, PCA on RGB, and combine into a new image (R=PC1, G=ExG, B=PC2).")
    parser.add_argument("-i", "--input", required=True, help="Path to the input image file.")
    parser.add_argument("-o", "--output", help="Path to save the output processed image. If omitted, saves next to input with '_processed' suffix.")

    args = parser.parse_args()

    # Determine output path if not provided
    output_file = args.output
    if not output_file:
        base, ext = os.path.splitext(args.input)
        output_file = f"{base}_processed{ext}"
        # Ensure output extension is suitable for images if input wasn't typical
        if ext.lower() not in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
             output_file = f"{base}_processed.png"


    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found at {args.input}")
    else:
        process_image(args.input, output_file)