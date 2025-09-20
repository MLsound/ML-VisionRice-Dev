# python apply_clahe_dataset.py
# ../../data/raw/3.5m.v4i.yolov8.640px/train/images
# ../../data/raw/3.5m.v4i.yolov8/valid/images
# ../../data/raw/3.5m.v4i.yolov8_clahe4/valid/images
# ../../data/processed/3.5m.v4i.yolov8.640px/valid/images
# ../../data/raw/3.5m.v3i.yolov8/valid/images
# ../../data/samples/3,5m90/Tests
# ../../data/final/3.5m.v4i.yolov8_blended.640px_clahe/train/images/
# ../../data/processed/3.5m.v4i.yolov8_blended.640px

# ../../data/raw/prueba/


import os
import cv2
from tqdm import tqdm
import time
history_time = []

def apply_clahe(image_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

    Args:
        image_path (str): The path to the image file.
        clip_limit (float, optional): Threshold for contrast limiting. Defaults to 2.0.
        tile_grid_size (tuple, optional): Size of grid for histogram equalization. Defaults to (8, 8).

    Returns:
        numpy.ndarray: The CLAHE processed image, or None if loading fails.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image at {image_path}")
            return None

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        if len(img.shape) == 2:  # Grayscale
            equalized_img = clahe.apply(img)
        elif len(img.shape) == 3:  # Color (BGR)
            # Apply CLAHE to each channel (you might want to experiment with other color spaces)
            b, g, r = cv2.split(img)
            cl_b = clahe.apply(b)
            cl_g = clahe.apply(g)
            cl_r = clahe.apply(r)
            equalized_img = cv2.merge((cl_b, cl_g, cl_r))
        else:
            print(f"Warning: Image at {image_path} has an unexpected number of channels ({img.shape[2]}). Returning original.")
            return img

        return equalized_img

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def apply_clahe_to_dataset(dataset_path, output_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies CLAHE to all images in a dataset and saves the results to a new directory.

    Args:
        dataset_path (str): The path to the directory containing the input images.
        output_path (str): The path to the directory where the CLAHE processed images will be saved.
        clip_limit (float, optional): Threshold for contrast limiting. Defaults to 2.0.
        tile_grid_size (tuple, optional): Size of grid for histogram equalization. Defaults to (8, 8).
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    image_files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'))]

    for image_file in tqdm(image_files, desc="Applying CLAHE"):
        # Start of loop turn
        start_time = time.perf_counter() # Mide tiempo  de ejecución
        
        input_image_path = os.path.join(dataset_path, image_file)
        output_image_path = os.path.join(output_path, image_file)

        equalized_img = apply_clahe(input_image_path, clip_limit, tile_grid_size)

        if equalized_img is not None:
            try:
                cv2.imwrite(output_image_path, equalized_img)
            except Exception as e:
                print(f"Error saving image {output_image_path}: {e}")
            
        # End of loop turn
        end_time = time.perf_counter() # Mide tiempo  de ejecución
        elapsed_time = end_time - start_time
        history_time.append(elapsed_time)
    print(f"\nCLAHE applied to all images in '{dataset_path}'. Processed images saved to '{output_path}'.")

def main():
    dataset_path = input("Enter the path to your input dataset directory: ")
    if not os.path.isdir(dataset_path):
        print(f"Error: Directory not found at {dataset_path}")
        return

    output_path = input("Enter the path for the output directory to save CLAHE processed images: ")

    clip_limit_str = input(f"Enter the CLAHE clip limit (default: 2.0): ")
    clip_limit = float(clip_limit_str) if clip_limit_str else 2.0

    tile_grid_size_str = input(f"Enter the CLAHE tile grid size (e.g., '8,8', default: 8,8): ")
    if tile_grid_size_str:
        try:
            tile_size = tuple(map(int, tile_grid_size_str.split(',')))
            if len(tile_size) == 2 and all(x > 0 for x in tile_size):
                tile_grid_size = tile_size
            else:
                print("Invalid tile grid size format. Using default (8, 8).")
        except ValueError:
            print("Invalid tile grid size format. Using default (8, 8).")
    else:
        tile_grid_size = (8, 8)
    
    apply_clahe_to_dataset(dataset_path, output_path, clip_limit, tile_grid_size)
    
    # At the end, after loop finishes
    average_time = sum(history_time) / len(history_time) if history_time else 0
    print(f"⏰ Tiempo promedio de procesamiento por archivo: {average_time:.3f} segundos")

if __name__ == "__main__":
    main()