from PIL import Image
import os
import sys


def convert_png_to_ico(input_png_path, output_ico_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, input_png_path) if not os.path.isabs(input_png_path) else input_png_path
    output_path = os.path.join(script_dir, output_ico_path) if not os.path.isabs(output_ico_path) else output_ico_path

    try:
        img = Image.open(input_path)
        img.save(output_path, format="ICO")
        print(f"Successfully converted '{input_path}' to '{output_path}'")
        return True
    except FileNotFoundError:
        print(f"Error: Input PNG file not found at '{input_path}'", file=sys.stderr)
        return False
    except Exception as e:
        print(f"An error occurred during conversion: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    convert_png_to_ico("logo_stasrg.png", "logo_stasrg.ico")