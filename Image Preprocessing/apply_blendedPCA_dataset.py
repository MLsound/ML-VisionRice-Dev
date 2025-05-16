# TESTING: python apply_blendedPCA_dataset.py -i '../../data/samples/3,5m90/'
# PROCESSING: python apply_blendedPCA_dataset.py -i '../../data/raw/datos_testing/test/images'
#'/Users/alejandrolloveras/Documents/ESTUDIO/UBA/Trabajo Final/Desarrollo/data/raw/
import colorPCA_BurnBlend
import argparse
import os

def batch_process_images(input_dir, output_dir):
    """
    Processes all images in a directory using process_image and color_burn_blend.

    Args:
        input_dir (str): Directory containing the input images.
        output_dir (str): Directory to save the processed images.
    """
    if input_dir[:-1] != '/':
        input_dir += '/'

    if output_dir == 'output':
       output_dir = f'{input_dir}output/'
    elif output_dir[:-1] != '/':
        output_dir += '/'
    print(f"Start processing '{input_dir}'...")
    print(f"Files will be stored into '{output_dir}'")
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    # Get all entries in the directory
    all_entries = os.listdir(input_dir)

    # Filter for image files and ensure they are actually files
    image_files_to_process = [
        f for f in all_entries
        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ]

    num_image_files = len(image_files_to_process)
    print(f"Found {num_image_files} image file(s) to process.")
    print("="*100) # Add a separator for clarity

    # Loop through the filtered list of image files
    # Removed the incorrect print statement `print(" - Files founded:",len(filename))`
    for i, filename in enumerate(image_files_to_process): # Added enumeration for optional progress print
        input_path = os.path.join(input_dir, filename)
        output_path = f'{output_dir}{filename}'
        
        print(f"{i+1}º Processing {filename}...")
        colorPCA_BurnBlend.process_image(input_path, output_path)
        print(f"✓ Finished processing {filename}")
        print("="*100)
    print(f"✅ Process complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process images in a directory.")
    parser.add_argument("-i", "--input_dir", required=True, help="Directory containing the input images.")
    parser.add_argument("-o", "--output_dir", default="output", help="Directory to save the processed images. Defaults to 'output'.")

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found at {args.input_dir}")
    else:
        batch_process_images(args.input_dir, args.output_dir)