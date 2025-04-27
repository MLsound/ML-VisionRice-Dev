from PIL import Image
import numpy as np

def color_burn_blend(base_img_path, blend_img_path, output_img_path):
    """
    Applies the Color Burn blend mode from Photoshop to two images.

    Args:
        base_img_path (str): Path to the base image file.
        blend_img_path (str): Path to the blend image file (the layer on top).
        output_img_path (str): Path to save the resulting image.
    """
    try:
        # Open images
        base_img = Image.open(base_img_path).convert("RGB")
        blend_img = Image.open(blend_img_path).convert("RGB")
    except FileNotFoundError:
        print("Error: Make sure the image paths are correct.")
        return
    except Exception as e:
        print(f"Error opening images: {e}")
        return

    # Ensure images have the same size
    if base_img.size != blend_img.size:
        print("Error: Images must have the same dimensions.")
        return

    # Convert images to NumPy arrays and normalize to [0, 1]
    base_np = np.array(base_img).astype(np.float32) / 255.0
    blend_np = np.array(blend_img).astype(np.float32) / 255.0

    # Initialize the output array
    output_np = np.zeros_like(base_np, dtype=np.float32)

    # Apply the Color Burn formula channel by channel
    # R = 1 - (1 - A) / B
    # Handle division by zero and the case where B is 1
    for c in range(3): # Iterate over R, G, B channels
        A = base_np[:, :, c]
        B = blend_np[:, :, c]

        # Create a mask for pixels where blend color is not zero
        non_zero_blend_mask = B > 0

        # Apply the formula only where blend color is not zero
        # For blend color = 0, the result is 0 (handled by initialization)
        output_np[non_zero_blend_mask, c] = 1.0 - (1.0 - A[non_zero_blend_mask]) / B[non_zero_blend_mask]

        # Clamp the result to the range [0, 1]
        output_np[:, :, c] = np.clip(output_np[:, :, c], 0.0, 1.0)

    # Convert back to 0-255 integer range
    output_np = (output_np * 255.0).astype(np.uint8)

    # Create a new Pillow image from the result array
    output_img = Image.fromarray(output_np, "RGB")

    # Save the output image
    try:
        output_img.save(output_img_path)
        print(f"Color Burn image saved successfully to {output_img_path}")
    except Exception as e:
        print(f"Error saving the output image: {e}")


if __name__ == "__main__":
    # --- Example Usage ---
    # Create dummy images for demonstration if you don't have your own
    # Replace these paths with your actual image paths

    # # Create a base image (gradient from black to white)
    # width, height = 400, 300
    # base_gradient = np.linspace(0, 255, width, dtype=np.uint8)
    # base_img_data = np.tile(base_gradient, (height, 1))
    # base_img_data = np.stack([base_img_data] * 3, axis=-1) # Make it RGB
    # dummy_base_img = Image.fromarray(base_img_data, "RGB")
    # dummy_base_img.save("base_image.png")

    # # Create a blend image (solid gray)
    # dummy_blend_img = Image.new("RGB", (width, height), color=(128, 128, 128)) # 50% gray
    # dummy_blend_img.save("blend_image.png")

    # Define input and output paths
    base_image_path = '/Users/alejandrolloveras/Documents/ESTUDIO/UBA/Trabajo Final/Desarrollo/data/samples/3,5m90/209_101_24.JPG'
    blend_image_path = '/Users/alejandrolloveras/Documents/ESTUDIO/UBA/Trabajo Final/Desarrollo/data/samples/3,5m90/Tests/Copia de 209_101_24_processed.JPG'
    output_image_path = '/Users/alejandrolloveras/Documents/ESTUDIO/UBA/Trabajo Final/Desarrollo/data/samples/3,5m90/Tests/BurnBlend/209_101_24_blended.JPG'

    # Apply the Color Burn blend mode
    color_burn_blend(base_image_path, blend_image_path, output_image_path)

    #print("\nDummy images 'base_image.png' and 'blend_image.png' created.")
    print(f"Resulting image saved as '{output_image_path}'.")
    #print("You can replace 'base_image.png' and 'blend_image.png' with your own images.")