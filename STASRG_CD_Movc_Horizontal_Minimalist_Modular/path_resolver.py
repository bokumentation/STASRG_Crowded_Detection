import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) 

def get_resource_path(relative_path):
    """
    Get absolute path to resource, works for development and PyInstaller.
    
    When PyInstaller bundles a file using '--add-data "file.pt;."', 
    it is placed at the root of the temporary bundle.
    """
    try:
        base_path = sys._MEIPASS
        # print(f"DEBUG: Running in EXE mode, Base Path: {base_path}")
    except Exception:
        base_path = PROJECT_ROOT
        # print(f"DEBUG: Running in DEV mode, Base Path: {base_path}")

    final_path = os.path.join(base_path, relative_path)
    return final_path
