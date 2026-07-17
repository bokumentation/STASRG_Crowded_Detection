import os
import sys

# Define the Project Root (The directory where app.py and the .pt file should be located)
# Since path_resolver.py is in the root, this is the correct definition.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) 

def get_resource_path(relative_path):
    """
    Get absolute path to resource, works for development and PyInstaller.
    
    When PyInstaller bundles a file using '--add-data "file.pt;."', 
    it is placed at the root of the temporary bundle.
    """
    try:
        # Running as an executable (PyInstaller bundle)
        # sys._MEIPASS points to the root of the temporary bundle (where your .pt file lives).
        base_path = sys._MEIPASS
        print(f"DEBUG: Running in EXE mode, Base Path: {base_path}")
    except Exception:
        # Running as a normal Python script (your development environment)
        # Base path is the project root (where the script is located).
        base_path = PROJECT_ROOT
        print(f"DEBUG: Running in DEV mode, Base Path: {base_path}")

    # Construct the final path (e.g., C:\temp\_MEIxxxx\survei2.pt)
    final_path = os.path.join(base_path, relative_path)
    return final_path

# We no longer need the get_model_file_path function, as the name is configured in app.py.
# The core functionality is now consolidated into get_resource_path.