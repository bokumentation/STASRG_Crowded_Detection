from PIL import Image
import os

def convert_png_to_ico(input_png_path, output_ico_path, sizes=None):
    """
    Converts a PNG image to ICO format.

    Args:
        input_png_path (str): The path to the input PNG file.
        output_ico_path (str): The path to save the output ICO file.
        sizes (list of tuples, optional): A list of (width, height) tuples
                                         for desired icon sizes.
                                         Defaults to Pillow's default sizes.
    """
    try:
        img = Image.open(input_png_path)
        if sizes:
            img.save(output_ico_path, format='ICO', sizes=sizes)
        else:
            img.save(output_ico_path, format='ICO')
        print(f"Successfully converted '{input_png_path}' to '{output_ico_path}'")
    except FileNotFoundError:
        print(f"Error: Input PNG file not found at '{input_png_path}'")
    except Exception as e:
        print(f"An error occurred during conversion: {e}")

if __name__ == "__main__":
    # Example usage:
    # Replace 'input.png' with your PNG file and 'output.ico' with desired output name
    input_file = "logo_stasrg.png"
    output_file = "output.ico"

    # Optional: Specify custom icon sizes (e.g., for favicon)
    # common_favicon_sizes = [(16, 16), (32, 32), (48, 48)]
    # convert_png_to_ico(input_file, output_file, sizes=common_favicon_sizes)

    # Convert with default sizes
    convert_png_to_ico(input_file, output_file)