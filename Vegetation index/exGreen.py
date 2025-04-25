# python colorPCA.py -i /path/to/your/input_image.jpg -o /path/to/save/result.png
# For testing: 
# '../../data/samples/3,5m90/209_101_24.JPG'
# '../../data/samples/3,5m90/503_118_29.JPG' 
# '../../data/samples/3,5m90/503_119_43.JPG'
# python exGreen.py -i '../../data/samples/3,5m90/503_119_43.JPG' -t p

# Setting applied for soil images (mask for removing plants):
# '/data/raw/soil/Soil1.jpg'
# '/data/raw/soil/Soil2.jpg'
# python exGreen.py -i '../../data/raw/soil/Soil2.jpg' -t p90 -dn -inv


import cv2
import numpy as np
import argparse
import os

def scale_channel(channel):
    """Scales a channel to 0-255 range for visualization."""
    # Perform min-max scaling
    return cv2.normalize(channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype('uint8')

def apply_threshold(channel, threshold: str | int | None):
    """Applies a threshold to a channel.

    Args:
        channel (np.ndarray): The input channel to apply the threshold on (expected 0-255 uint8).
        threshold (str | int | None): The threshold value. Can be:
            - An integer (0-255) for absolute thresholding.
            - A string starting with 'p' followed by a number (e.g., 'p90') for percentile-based thresholding.
            - None, in which case no thresholding is applied.

    Returns:
        np.ndarray: The binary image after applying the threshold (0 or 255 values) or the original channel if threshold is None.
    """
    if threshold is None:
        print("No thresholding applied.")
        return channel # Return the original channel if no threshold

    if isinstance(threshold, str) and threshold.startswith('p'):
        try:
            percentile = int(threshold[1:])
            # Calculate dynamic threshold based on the specified percentile
            # Use the scaled channel (0-255) for percentile calculation
            threshold_value = np.percentile(channel, percentile)
            print(f"Threshold calculated based on percentile ({percentile}): {threshold_value}")
        except ValueError:
            raise ValueError("The 'threshold' value can either be an absolute value (0-255) or indicate a percentile, starting with 'p' followed by the percentile (0-100) (e.g., 'p90' = 90th percentile).")
    else:
        threshold_value = int(threshold)
        print(f"Applying absolute threshold: {threshold_value}")

    # Apply threshold
    # Use cv2.THRESH_BINARY_EXG? No, THRESH_BINARY is standard
    _, binary = cv2.threshold(channel, threshold_value, 255, cv2.THRESH_BINARY)

    return binary

def apply_gaussian_blur(image, kernel_size):
    """Applies Gaussian blur to a grayscale image."""
    if kernel_size <= 0 or kernel_size % 2 == 0:
        print("Warning: Gaussian blur kernel size must be a positive odd integer. Skipping Gaussian blur.")
        return image
    print(f"Applying Gaussian blur with kernel size: {kernel_size}")
    # Kernel size must be positive and odd
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_morphological_ops(image, kernel_size):
    """Applies morphological opening and closing to a binary image."""
    if kernel_size <= 0 or kernel_size % 2 == 0:
         print("Warning: Morphological kernel size must be a positive odd integer. Skipping morphological operations.")
         return image

    # Morphological operations should only be applied to binary images (values 0 or 255)
    # Check if the image looks binary (contains mostly 0 and 255 values)
    unique_values = np.unique(image)
    if not np.all(np.isin(unique_values, [0, 255])):
         print("Warning: Morphological operations are best applied to binary images. Input image may not be binary.")

    print(f"Applying morphological opening and closing with kernel size: {kernel_size}")
    # Create a elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Apply opening (removes small objects)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # Apply closing (fills small holes)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    return closing


def process_image(input_path, output_path, threshold=None, inverse=False, use_denoiser=False):
    """
    Reads an image, calculates ExG, applies optional denoising, threshold, and inverse.

    Args:
        input_path (str): Path to the input image file.
        output_path (str): Path to save the processed image file.
        threshold (float or str): Threshold value or percentile for ExG. If provided, applies thresholding.
        inverse (bool): If True, inverts the binary image (white becomes black and vice versa).
        use_denoiser (bool): If True, applies predefined Gaussian and Morphological denoising.

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
    processed_img = scale_channel(exg)
    print("ExG calculation complete.")

    # --- Apply Denoising if requested ---
    if use_denoiser:
        print("-" * 20)
        print("APPLYING DENOISER")
        print("-" * 20)
        # Apply Gaussian Blur (hardcoded ksize 7) to the grayscale ExG image
        gaussian_ksize = 7
        processed_img = apply_gaussian_blur(processed_img, gaussian_ksize)
        print("Gaussian blur applied.")

        # Apply Morphological Operations (hardcoded ksize 5)
        # This should ideally be applied *after* thresholding, as it works on binary images.
        # We will apply it here, but note that applying morphology to a grayscale image
        # might not have the intended effect of cleaning up binary noise.
        # A more typical pipeline would be: ExG -> Gaussian -> Threshold -> Morphology -> Inverse
        # However, following the implied structure of applying denoiser *before* threshold in the previous attempt,
        # let's reconsider the pipeline. The common use case is to denoise the *binary* mask.
        # Let's adjust the flow: ExG -> [Gaussian] -> [Threshold] -> [Morphology] -> [Inverse]

    # --- 2. Apply threshold if specified ---
    is_binary = False
    if threshold is not None:
        processed_img = apply_threshold(processed_img, threshold)
        print("Thresholding complete.")
        is_binary = True # Image is now binary

    # --- Apply Morphological Ops if denoiser used and threshold was applied ---
    if use_denoiser and is_binary:
        morph_ksize = 5
        processed_img = apply_morphological_ops(processed_img, morph_ksize)
        print("Morphological operations applied.")

    # --- 3. Apply inverse if specified ---
    if inverse:
        if not is_binary:
             print("Warning: Inverse applied to a non-binary image (threshold was not used).")
        print("Inversion complete.")
        processed_img = cv2.bitwise_not(processed_img)

    # --- 4. Save the output image ---
    print(f"Saving processed image to: {output_path}")
    try:
        cv2.imwrite(output_path, processed_img)
        print("Image saved successfully.")
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False

# --- Main execution block ---
from typing import Union
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image: Calculate ExG, apply optional denoising, threshold, and inverse.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input image file.")
    parser.add_argument("-o", "--output", help="Path to save the output processed image. If omitted, the output file will be saved in the same directory as the input file with '_processed' appended to the filename.")
    parser.add_argument("-t", "--threshold", type=str, help="The 'threshold' value can either be an absolute value (0-255) or indicate a percentile, starting with 'p' followed by the percentile (0-100) (e.g., 'p90' = 90th percentile).")
    parser.add_argument("-inv", "--inverse", action="store_true", help="If specified, inverts the binary image (white becomes black and vice versa). Applies after thresholding.")
    parser.add_argument("-dn", "--denoiser", action="store_true", help="If specified, applies predefined Gaussian (k=7) and Morphological (k=5) denoising.")
    args = parser.parse_args()

    # Determine output path if not provided
    output_file = args.output
    if not output_file:
        base, ext = os.path.splitext(args.input)
        # Append '_processed' and potentially '_denoised' to the filename base
        output_suffix = "_processed"
        if args.denoiser:
            output_suffix += "_denoised"
        output_file = f"{base}{output_suffix}{ext}"
        # Ensure output extension is suitable for images if input wasn't typical
        if ext.lower() not in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
             output_file = f"{base}{output_suffix}.png"


    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found at {args.input}")
    else:
        process_image(args.input, output_file,
                      threshold=args.threshold,
                      inverse=args.inverse,
                      use_denoiser=args.denoiser)