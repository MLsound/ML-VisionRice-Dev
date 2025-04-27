"""
Summary:
This script processes a source folder containing image files and creates a fake dataset for data augmentation. 
It copies all `.jpg` or `.jpeg` images from the source folder to a destination folder for images and creates 
corresponding empty `.txt` label files in a separate destination folder for labels.

Functions:
- Copies image files from the source directory to the destination images directory.
- Creates empty `.txt` label files in the destination labels directory with the same base name as the images.

Usage:
- Define the `source_folder` path containing the images.
- Define the `dest_images_folder` and `dest_labels_folder` paths for the output dataset.
- Run the script to process the images and generate the fake dataset.

Variables:
- `source_folder`: Path to the source directory containing the images.
- `dest_images_folder`: Path to the destination directory for copied images.
- `dest_labels_folder`: Path to the destination directory for empty label files.
- `copied_count`: Counter for the number of images successfully copied.
- `label_created_count`: Counter for the number of label files successfully created.

Error Handling:
- Handles `IOError` during file operations and logs the error.
- Catches unexpected exceptions and logs them for debugging purposes.
"""
import os
import shutil

# Define source and destination directories
source_folder = '../../data/processed/soil'
dest_images_folder = '../../data/processed/soil_aug/train/images'
dest_labels_folder = '../../data/processed/soil_aug/train/labels'

# Create destination directories if they don't exist
os.makedirs(dest_images_folder, exist_ok=True)
os.makedirs(dest_labels_folder, exist_ok=True)

print(f"Source folder: {source_folder}")
print(f"Destination images folder: {dest_images_folder}")
print(f"Destination labels folder: {dest_labels_folder}")
print("-" * 30)

copied_count = 0
label_created_count = 0

# Walk through the source folder and its subdirectories
for root, _, files in os.walk(source_folder):
    for file in files:
        # Check if the file is a JPG image (case-insensitive)
        if file.lower().endswith('.jpg') or file.lower().endswith('.jpeg'):
            source_file_path = os.path.join(root, file)
            dest_image_path = os.path.join(dest_images_folder, file)

            try:
                # Copy the image file
                shutil.copy2(source_file_path, dest_image_path)
                copied_count += 1
                print(f"Copied: {source_file_path} to {dest_image_path}")

                # Create an empty .txt file with the same name in the labels folder
                label_filename = os.path.splitext(file)[0] + '.txt'
                dest_label_path = os.path.join(dest_labels_folder, label_filename)

                with open(dest_label_path, 'w') as f:
                    pass # Create an empty file

                label_created_count += 1
                print(f"Created empty label file: {dest_label_path}")

            except IOError as e:
                print(f"Error processing file {source_file_path}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred with file {source_file_path}: {e}")


print("-" * 30)
print(f"Script finished.")
print(f"Total images copied: {copied_count}")
print(f"Total label files created: {label_created_count}")