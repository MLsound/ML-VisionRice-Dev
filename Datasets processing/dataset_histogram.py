# dataset_histogram.py
# ../../data/raw/3.5m.v3i.yolov8/train/images
# ../../data/raw/3.5m.v3i.yolov8/valid/images
# ../../data/samples/3,5m90/Tests

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def calculate_channel_histogram(image_path, channel=0):
    """
    Calculates the histogram for a specific color channel of an image.

    Args:
        image_path (str): The path to the image file.
        channel (int): The channel index (0 for Blue, 1 for Green, 2 for Red)
                       for color images, or 0 for grayscale. Defaults to 0.

    Returns:
        tuple: A tuple containing the histogram array and the bin edges,
               or None if the image cannot be loaded or the channel is invalid.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image at {image_path}")
            return None

        if len(img.shape) == 2:  # Grayscale
            if channel != 0:
                print(f"Warning: Image at {image_path} is grayscale. Ignoring channel {channel}, using channel 0.")
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            return hist.flatten(), np.arange(257)
        elif img.shape[2] == 3:  # Color (BGR)
            if 0 <= channel <= 2:
                hist = cv2.calcHist([img], [channel], None, [256], [0, 256])
                return hist.flatten(), np.arange(257)
            else:
                print(f"Error: Invalid channel index ({channel}) for color image at {image_path}. Choose 0, 1, or 2.")
                return None
        else:
            print(f"Warning: Image at {image_path} has an unexpected number of channels ({img.shape[2]}).")
            return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def calculate_dataset_histograms(dataset_path, save_path="dataset_histograms"):
    """
    Calculates and plots the average histogram for each color channel
    across all images in a dataset.

    Args:
        dataset_path (str): The path to the directory containing the images.
        save_path (str, optional): The path to the directory where the histogram
                                    plots will be saved. Defaults to "dataset_histograms".
    """
    image_files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'))]
    if not image_files:
        print(f"No image files found in the directory: {dataset_path}")
        return

    all_red_hists = []
    all_green_hists = []
    all_blue_hists = []
    all_gray_hists = []
    is_color_dataset = None

    os.makedirs(save_path, exist_ok=True)

    for image_file in tqdm(image_files, desc="Processing Images"):
        image_path = os.path.join(dataset_path, image_file)
        img = cv2.imread(image_path)

        if img is not None:
            if len(img.shape) == 2:  # Grayscale
                is_color_dataset = False if is_color_dataset is None else is_color_dataset
                hist, _ = calculate_channel_histogram(image_path, 0)
                if hist is not None:
                    all_gray_hists.append(hist)
            elif img.shape[2] == 3:  # Color
                is_color_dataset = True if is_color_dataset is None else is_color_dataset
                red_hist, _ = calculate_channel_histogram(image_path, 2)
                green_hist, _ = calculate_channel_histogram(image_path, 1)
                blue_hist, _ = calculate_channel_histogram(image_path, 0)
                if red_hist is not None and green_hist is not None and blue_hist is not None:
                    all_red_hists.append(red_hist)
                    all_green_hists.append(green_hist)
                    all_blue_hists.append(blue_hist)
            else:
                print(f"Skipping image {image_file} with unexpected number of channels.")

    if is_color_dataset is None and not all_gray_hists:
        print("No valid images found to calculate histograms.")
        return

    plt.figure(figsize=(10, 6))
    bins = np.arange(256)

    if is_color_dataset is True and all_red_hists and all_green_hists and all_blue_hists:
        avg_red_hist = np.mean(all_red_hists, axis=0)
        avg_green_hist = np.mean(all_green_hists, axis=0)
        avg_blue_hist = np.mean(all_blue_hists, axis=0)

        plt.plot(bins, avg_red_hist, color='red', alpha=0.7, label='Average Red')
        plt.plot(bins, avg_green_hist, color='green', alpha=0.7, label='Average Green')
        plt.plot(bins, avg_blue_hist, color='blue', alpha=0.7, label='Average Blue')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Average Frequency')
        plt.title('Average Color Histograms of the Dataset')
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.savefig(os.path.join(save_path, 'average_color_histogram.png'))
        plt.close()
        print(f"Average color histogram saved to {os.path.join(save_path, 'average_color_histogram.png')}")

    elif is_color_dataset is False and all_gray_hists:
        avg_gray_hist = np.mean(all_gray_hists, axis=0)
        plt.plot(bins, avg_gray_hist, color='gray', alpha=0.7, label='Average Gray')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Average Frequency')
        plt.title('Average Grayscale Histogram of the Dataset')
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.savefig(os.path.join(save_path, 'average_grayscale_histogram.png'))
        plt.close()
        print(f"Average grayscale histogram saved to {os.path.join(save_path, 'average_grayscale_histogram.png')}")

    elif is_color_dataset is None and not all_gray_hists:
        print("No histograms were calculated.")

def main():
    dataset_path = input("Enter the path to your dataset directory: ")
    if not os.path.isdir(dataset_path):
        print(f"Error: Directory not found at {dataset_path}")
        return

    save_path = input("Enter the path to save the histogram plots (default: dataset_histograms): ") or "dataset_histograms"

    calculate_dataset_histograms(dataset_path, save_path)

if __name__ == "__main__":
    main()