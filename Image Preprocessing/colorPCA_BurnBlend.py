# python colorPCA_BurnBlend.py -i /path/to/your/input_image.jpg -o /path/to/save/result.png
# For testing: 
# '../../data/samples/3,5m90/209_101_24.JPG'
# '../../data/samples/3,5m90/503_118_29.JPG' 
# '../../data/samples/3,5m90/503_119_43.JPG'
# python colorPCA_BurnBlend.py -i '../../data/samples/3,5m90/503_119_43.JPG'
# python colorPCA_BurnBlend.py -i '../../data/raw/CLAHE/209_205_50_JPG.rf.dcf6a84732d63af4945585a50533cd01.jpg'

import colorPCA
import BurnBlend
import argparse
import os

def create_dir(path):
    try:
        os.makedirs(path)
        print(f"Created output directory: {path}")
    except Exception as e:
        print(f"Error creating output directory {path}: {e}")

def process_image(input_path, output_file, output_dir = 'BurnBlend', save_pca = False):
    
    if output_dir is None:
        defined_output_dir = False
        output_dir = 'BurnBlend'
    else:
        defined_output_dir = True
    
    # Defining and creating output dir
    if not output_file:
        # Extracts info about file and folders
        base, ext = os.path.splitext(input_path)
        folders = base.split('/')
        filename = folders.pop()

        # Assign output folder
        if output_dir is not None and output_dir != '': folders.insert(len(folders), output_dir)
        #print(folders)
        base = '/'.join(folders)
    else:
        # Extracts info about file and folders
        base, ext = os.path.splitext(output_file)
        folders = base.split('/')
        filename = folders.pop()
        base = '/'.join(folders)
        if defined_output_dir:
            print('Alert: As long as an "--output" value is defined, "--folder" setting will be omited.')

    # Ensure output extension is suitable for images if input wasn't typical
    if ext.lower() not in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
        ext = '.png'
        print(f'Warning: extension is not defined. Setted by default as {ext}')
        
    # Creating filenames
    output_pca_file = f"{base}/{filename}_processed{ext}"
    output_blend_file = f"{base}/{filename}_blended{ext}"
    #print(folders, filename)

    # Check if output_dir exists, if not create the folder
    if base:
        if not os.path.exists(base):
            create_dir(base)
    else:
        raise ValueError ('Output dir is not defined.')

    # Applies PCA to color channels
    colorPCA.process_image(input_path, output_pca_file)

    # Applies Burn Blend between colorPCA & original image
    BurnBlend.color_burn_blend(input_path, output_pca_file, output_blend_file)
    # It works as a mask for augmenting saturation of plants
    # while reducing brightness of background (soil)

    # Delete of PCA file (if not needed)
    if not save_pca: 
        try:
            os.remove(output_pca_file)
            print(f"Intermediate PCA file {output_pca_file} deleted.")
        except Exception as e:
            print(f"Error deleting PCA file: {e}")


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image: Perform PCA on RGB channels, calculate ExG, and apply Color Burn blend mode.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input image file. This is the image that will be processed to calculate ExG, perform PCA on RGB channels, and apply the Color Burn blend mode.")
    parser.add_argument("-o", "--output", help="Path to save the output processed image. If omitted, saves next to input with '_processed' suffix.")
    parser.add_argument("-f", "--folder", help="Folder to save the output processed image. Just used when output is omited.")
    parser.add_argument("-pca", "--save-pca", action="store_true", help="Flag to save the intermediate PCA-processed image.")
    args = parser.parse_args()
    # if args.folder:
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found at {args.input}")
    else:
        process_image(args.input, args.output, args.folder, args.save_pca)


# python colorPCA_BurnBlend.py -i '../../data/samples/3,5m90/503_119_43.JPG'