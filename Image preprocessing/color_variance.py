# python color_variance.py
# ../../data/raw/3.5m.v3i.yolov8/train/images
# ../../data/raw/3.5m.v3i.yolov8/valid/images
# ../../data/samples/3,5m90/Tests

import os
import cv2
import numpy as np
from tqdm import tqdm

def calculate_color_variance(image_path):
    """
    Calculates the variance of each color channel (R, G, B) in an image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        dict: A dictionary containing the variance of each color channel
              ('red_variance', 'green_variance', 'blue_variance').
              Returns None if the image cannot be loaded.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image at {image_path}")
            return None

        # Ensure the image is in BGR format (OpenCV default)
        if len(img.shape) == 2:  # Grayscale image
            print(f"Warning: Image at {image_path} is grayscale. Returning variance of the single channel.")
            variance = np.var(img)
            return {'gray_variance': variance}
        elif img.shape[2] == 3:  # Color image
            blue, green, red = cv2.split(img)
            red_variance = np.var(red)
            green_variance = np.var(green)
            blue_variance = np.var(blue)
            return {
                'red_variance': red_variance,
                'green_variance': green_variance,
                'blue_variance': blue_variance
            }
        else:
            print(f"Warning: Image at {image_path} has an unexpected number of channels ({img.shape[2]}).")
            return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def measure_dataset_color_variance(dataset_path):
    """
    Calculates the average color variance across all images in a dataset.

    Args:
        dataset_path (str): The path to the directory containing the images.

    Returns:
        dict: A dictionary containing the average variance for each color channel
              ('avg_red_variance', 'avg_green_variance', 'avg_blue_variance')
              or ('avg_gray_variance') for grayscale datasets.
              Returns None if no valid images are found.
    """
    image_files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'))]
    if not image_files:
        print(f"No image files found in the directory: {dataset_path}")
        return None

    all_variances = []
    for image_file in tqdm(image_files, desc="Processing Images"):
        image_path = os.path.join(dataset_path, image_file)
        variances = calculate_color_variance(image_path)
        if variances:
            all_variances.append(variances)

    if not all_variances:
        print("No valid image variances were calculated.")
        return None

    # Calculate the average variance for each channel
    avg_variances = {}
    if 'red_variance' in all_variances[0]:
        avg_red_variance = np.mean([v['red_variance'] for v in all_variances])
        avg_green_variance = np.mean([v['green_variance'] for v in all_variances])
        avg_blue_variance = np.mean([v['blue_variance'] for v in all_variances])
        avg_variances = {
            'avg_red_variance': avg_red_variance,
            'avg_green_variance': avg_green_variance,
            'avg_blue_variance': avg_blue_variance
        }
    elif 'gray_variance' in all_variances[0]:
        avg_gray_variance = np.mean([v['gray_variance'] for v in all_variances])
        avg_variances = {'avg_gray_variance': avg_gray_variance}
    else:
        print("Unexpected variance dictionary structure.")
        return None

    return avg_variances

def main():
    dataset_path = input("Enter the path to your dataset directory: ")
    if not os.path.isdir(dataset_path):
        print(f"Error: Directory not found at {dataset_path}")
        return

    average_variances = measure_dataset_color_variance(dataset_path)

    if average_variances:
        print("\nAverage Color Variance of the Dataset:")
        for channel, variance in average_variances.items():
            print(f"{channel}: {variance:.4f}")

if __name__ == "__main__":
    main()