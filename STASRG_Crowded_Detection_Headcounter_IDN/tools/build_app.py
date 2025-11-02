import subprocess
import shlex
import os
import re
import glob
import shutil

# --- Dynamic Configuration (UPDATED FOR 'tools' FOLDER) ---
# 1. Determine the script's location (e.g., .../Project/tools)
SCRIPT_LOCATION = os.path.dirname(os.path.abspath(__file__))

# 2. Define the Project Root (The parent directory of the script's location)
PROJECT_ROOT = os.path.dirname(SCRIPT_LOCATION)

# 3. Define the main Python script path
#    Assumes main script is always 'app.py' in the PROJECT_ROOT.
main_script_name = "app.py"
main_script_path = os.path.join(PROJECT_ROOT, main_script_name)

# 4. Define the final output path (subfolder 'output' inside PROJECT_ROOT)
FINAL_DIST_PATH = os.path.join(PROJECT_ROOT, "output")

# 5. Define the Base Executable Name (Uses the PROJECT_ROOT folder name)
base_exe_name = os.path.basename(PROJECT_ROOT)

# 6. Find the *.pt file dynamically (Looks in the PROJECT_ROOT)
try:
    # Look for *.pt in the PROJECT_ROOT
    pt_file_path = glob.glob(os.path.join(PROJECT_ROOT, "*.pt"))[0]
    pt_file_name = os.path.basename(pt_file_path)
    print(f"✅ Found model file: {pt_file_name}")
except IndexError:
    print("❌ ERROR: Could not find any '*.pt' model file in the project root!")
    exit(1)

# 7. List of Data Files/Directories to include (Paths are relative to PROJECT_ROOT)
data_files = [
    f'{pt_file_path};.',                                    # The *.pt file
    os.path.join(PROJECT_ROOT, 'static') + ';static/',      # The static folder
    os.path.join(PROJECT_ROOT, 'templates') + ';templates/', # The templates folder
]

# 8. Define Icon Path (Assumes it's in the same 'tools' folder as the script)
icon_path = os.path.join(SCRIPT_LOCATION, 'output.ico') # Uses SCRIPT_LOCATION here


# --- Build Numbering Logic (Unchanged) ---

def get_next_build_name(base_name, distribution_path):
    """Checks the distribution directory and returns the next sequential build name."""
    if not os.path.exists(distribution_path):
        return f"{base_name}_1"

    pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)$")
    highest_number = 0

    for item in os.listdir(distribution_path):
        match = pattern.match(item)
        if match:
            current_number = int(match.group(1))
            if current_number > highest_number:
                highest_number = current_number

    next_number = highest_number + 1
    
    if next_number == 1 and os.path.exists(os.path.join(distribution_path, base_name)):
         print(f"⚠️ Warning: Found an unnumbered build '{base_name}' in the output path. New builds will be numbered starting at {next_number}.")

    return f"{base_name}_{next_number}"

# Determine the final name for the current build
current_exe_name = get_next_build_name(base_exe_name, FINAL_DIST_PATH)

# --- Build the PyInstaller Command ---
command_parts = [
    "pyinstaller",
    "--noconfirm",  
    "--onedir",
    "--console",
    f'--icon "{icon_path}"',
    f'--name "{current_exe_name}"',
    f'--distpath "{FINAL_DIST_PATH}"'
]

for data in data_files:
    command_parts.append(f'--add-data "{data}"')

command_parts.append(f'"{main_script_path}"')

pyinstaller_command = " ".join(command_parts)

# --- Execute the Command ---
print(f"\n🚀 Starting PyInstaller Build for Project: **{base_exe_name}**")
print(f"   Building as: **{current_exe_name}**")
print("-" * 70)
print(f"Output Directory: {FINAL_DIST_PATH}")
print("-" * 70)

try:
    subprocess.run(
        shlex.split(pyinstaller_command),
        check=True, 
        capture_output=False, 
        text=True
    )
    
    print("\n✅ PyInstaller build completed successfully!")
    print(f"Final EXE Path: **{os.path.join(FINAL_DIST_PATH, current_exe_name, current_exe_name + '.exe')}**")
    
    # --- CLEANUP LOGIC ---
    print("\n🧹 Cleaning up temporary files...")

    # 1. Remove the 'build' folder (located in the PROJECT_ROOT)
    build_path = os.path.join(PROJECT_ROOT, 'build')
    if os.path.exists(build_path):
        shutil.rmtree(build_path)
        print(f"   -> Removed temporary build folder: {build_path}")

    # 2. Remove the '.spec' file (located in the PROJECT_ROOT)
    spec_file = f"{base_exe_name}.spec"
    spec_path = os.path.join(PROJECT_ROOT, spec_file)
    if os.path.exists(spec_path):
        os.remove(spec_path)
        print(f"   -> Removed spec file: {spec_path}")
    
    print("   -> Cleanup complete. Project directory is tidy!")
    # --- END CLEANUP LOGIC ---

except subprocess.CalledProcessError as e:
    print(f"\n❌ PyInstaller build failed with error (Return Code {e.returncode}):")
    print("Please check the PyInstaller output above for details.")
except FileNotFoundError:
    print("\n❌ Error: PyInstaller command not found.")
    print("Please ensure 'pyinstaller' is installed and accessible in your system's PATH.")
except Exception as e:
    print(f"\n❌ An unexpected error occurred: {e}")