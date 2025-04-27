"""
merge_datasets.py

This script merges two datasets into a single destination folder. It handles the merging of 
dataset splits (train, test, valid) and their respective content directories (images, labels). 
The script ensures that files from the second dataset do not overwrite existing files from 
the first dataset. Additionally, it copies a YAML configuration file from the first dataset 
to the destination folder.

Steps performed by the script:
1. Checks if the destination folder exists and creates it if necessary.
2. Copies the YAML configuration file from the first dataset to the destination folder.
3. Copies the contents of the first dataset (train/test/valid splits and images/labels).
4. Copies the contents of the second dataset, ensuring no overwrites of existing files.

Usage:
- Update the `dataset1`, `dataset2`, and `dest_folder` variables with the appropriate paths.
- Run the script to merge the datasets.

Note:
- If the destination folder already exists, the script will terminate with an error message.
- The script assumes that the datasets follow a specific folder structure with splits and content directories.
"""

import os
import shutil
import sys

# Define source and destination folders
#dataset1 = '../../data/processed/3.5m.v3i.yolov8.640px/'
dataset1 = '../../data/processed/3.5m.v3i.yolov8.640px.aug.v1'
dataset2 = '../../data/processed/soil_aug/'
#dest_folder = '../../data/processed/3.5m.v3i.yolov8.640px.soil_aug/'
dest_folder = '../../data/processed/3.5m.v3i.yolov8.640px.aug.v1.soil_aug/'

# Define the standard subfolders for dataset splits and content
splits = ['train', 'test', 'valid']
content_dirs = ['images', 'labels']

print(f"Attempting to merge datasets:")
print(f"  Dataset 1: {dataset1}")
print(f"  Dataset 2: {dataset2}")
print(f"  Destination: {dest_folder}")
print("-" * 50)

# --- Step 1: Check for destination folder existence and create if necessary ---
if os.path.exists(dest_folder):
    print(f"Error: Destination folder '{dest_folder}' already exists.")
    print("Please remove it manually before running the script again.")
    sys.exit(1) # Exit the script with an error code
else:
    print(f"Destination folder '{dest_folder}' does not exist. Creating...")
    try:
        os.makedirs(dest_folder)
        print("Destination folder created successfully.")
    except OSError as e:
        print(f"Error creating destination folder {dest_folder}: {e}")
        sys.exit(1)

print("-" * 50)

# --- Step 2: Copy YAML file from dataset 1 ---
print(f"Copying YAML file from Dataset 1: {dataset1}")
yaml_file_copied = False
try:
    # Find the first YAML file in the root of dataset1
    for item in os.listdir(dataset1):
        if item.lower().endswith('.yaml'):
            src_yaml_path = os.path.join(dataset1, item)
            dest_yaml_path = os.path.join(dest_folder, item)
            shutil.copy2(src_yaml_path, dest_yaml_path)
            print(f"Copied YAML: {src_yaml_path} to {dest_yaml_path}")
            yaml_file_copied = True
            break # Assuming there's only one main YAML file
    if not yaml_file_copied:
        print(f"Warning: No .yaml file found in the root of {dataset1}")

except FileNotFoundError:
    print(f"Error: Dataset 1 path not found: {dataset1}")
    # Clean up the created destination folder before exiting
    if os.path.exists(dest_folder):
         print(f"Cleaning up incomplete destination folder: {dest_folder}")
         shutil.rmtree(dest_folder)
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while copying YAML file: {e}")
    if os.path.exists(dest_folder):
         print(f"Cleaning up incomplete destination folder: {dest_folder}")
         shutil.rmtree(dest_folder)
    sys.exit(1)

print("-" * 50)

# --- Step 3: Copy dataset 1 contents (train/test/valid + images/labels) ---
print(f"Copying contents from Dataset 1: {dataset1}")
for split in splits:
    src_split_folder = os.path.join(dataset1, split)
    dest_split_folder = os.path.join(dest_folder, split)

    if os.path.exists(src_split_folder):
        print(f"Copying split: {split} from {src_split_folder}")
        try:
            # Use copytree to copy the entire split folder and its contents
            shutil.copytree(src_split_folder, dest_split_folder)
            print(f"Successfully copied {split}.")
        except Exception as e:
            print(f"Error copying split {split} from dataset 1: {e}")
            # Clean up the created destination folder before exiting
            if os.path.exists(dest_folder):
                print(f"Cleaning up incomplete destination folder: {dest_folder}")
                shutil.rmtree(dest_folder)
            sys.exit(1)
    else:
        print(f"Warning: Split folder not found in Dataset 1: {src_split_folder}")

print("-" * 50)

# --- Step 4: Copy dataset 2 contents (train/test/valid + images/labels) ---
# Files from dataset2 will overwrite if they don't exist in dest_folder (i.e., weren't in dataset1)
print(f"Copying contents from Dataset 2: {dataset2} (handling potential duplicates)")
for split in splits:
    src_split_folder = os.path.join(dataset2, split)
    dest_split_folder = os.path.join(dest_folder, split) # This should already exist from dataset1 copy

    if os.path.exists(src_split_folder):
        print(f"Processing split: {split} from {dataset2}")
        for content_dir in content_dirs:
            src_content_folder = os.path.join(src_split_folder, content_dir)
            dest_content_folder = os.path.join(dest_split_folder, content_dir) # This should already exist

            if os.path.exists(src_content_folder):
                 # Ensure the destination content folder exists (though copytree should have created it)
                 os.makedirs(dest_content_folder, exist_ok=True)
                 print(f"  Copying {content_dir} from {src_content_folder}")

                 # Walk through the source content folder (dataset2)
                 for root, _, files in os.walk(src_content_folder):
                      # Calculate the relative path from the source content folder root
                      # This is needed to recreate the sub-directory structure in the destination
                      rel_path = os.path.relpath(root, src_content_folder)
                      current_dest_folder = os.path.join(dest_content_folder, rel_path)

                      # Ensure the corresponding destination sub-folder exists
                      os.makedirs(current_dest_folder, exist_ok=True)

                      for file in files:
                           src_file_path = os.path.join(root, file)
                           dest_file_path = os.path.join(current_dest_folder, file)

                           # Check if the file already exists in the destination (from dataset1)
                           if os.path.exists(dest_file_path):
                               print(f"    Warning: File already exists in destination, skipping: {dest_file_path}")
                               # Omit the file from dataset 2
                           else:
                                try:
                                    shutil.copy2(src_file_path, dest_file_path)
                                    # print(f"    Copied: {src_file_path} to {dest_file_path}") # Uncomment for verbose output
                                except Exception as e:
                                     print(f"    Error copying file {src_file_path}: {e}")
            else:
                 print(f"  Warning: Content folder not found in Dataset 2: {src_content_folder}")
    else:
        print(f"Warning: Split folder not found in Dataset 2: {src_split_folder}")


print("-" * 50)
print("Dataset merging script finished.")
print(f"Merged dataset created at: {dest_folder}")