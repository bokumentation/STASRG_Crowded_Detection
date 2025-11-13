import webbrowser
import subprocess
import time
import os
import json
from flask import Flask, Response, render_template, jsonify, send_file, request
import cv2
from io import BytesIO
from openpyxl import Workbook
import threading
from config_writer import save_config_to_file
import numpy as np

# Import the two processor classes
from cv_processor_vertical import VerticalProcessor
from cv_processor_horizontal import HorizontalProcessor
from datetime import datetime
from path_resolver import get_resource_path

# --- CONFIGURATION LOADING AND PROCESSOR SELECTION ---
CONFIG_FILE_NAME = "config.json"


def load_config():
    """Load configuration from the specified JSON file."""
    try:
        # Use resolver to find the absolute path of config.json
        absolute_config_path = get_resource_path(CONFIG_FILE_NAME) 
        
        # Load the configuration
        with open(absolute_config_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"FATAL ERROR: Could not load or parse config.json: {e}")
        return None


CONFIG = load_config()

if CONFIG is None:
    print("Application cannot start without a valid config.json. Exiting.")
    exit(1)

VERSION = CONFIG.get("version", "horizontal").lower()  # Default to horizontal

print(f" ")
print(f" ")
print(f"---   Memulai Aplikasi Crowded Detection   ---")
print(f" ")
print(f" ")
print(f"Versi: {VERSION.upper()}")

# Dynamically select and instantiate the processor
if VERSION == "vertical":
    ProcessorClass = VerticalProcessor
    DIMENSION_KEY = "vertical_lines" # Key to access line positions
    print("Versi: Vertical Line Counting.")
elif VERSION == "horizontal":
    ProcessorClass = HorizontalProcessor
    DIMENSION_KEY = "horizontal_lines" # Key to access line positions
    print("Versi: Horizontal Line Counting.")
else:
    print(f"ERROR: Unknown version '{VERSION}' in config.json. Using Horizontal.")
    ProcessorClass = HorizontalProcessor
    DIMENSION_KEY = "horizontal_lines"


try:
    processor = ProcessorClass(config=CONFIG)
    if processor.model is None or processor.cap is None:
        raise RuntimeError("CV Processor gagal inisialisasi model atau camera.")
except Exception as e:
    print(f"FATAL ERROR: GAGAL MENJALANKAN APLIKASI. PERIKSA KONEKSI KAMERA: {e}")
    processor = None
    exit(1)

app = Flask(__name__, static_folder="static")


def generate_frames():
    """Generator function to stream video frames (Motion JPEG)."""
    while True:
    # Capture frame, success flag, and NEW status flag
        frame, ret, status = processor.get_frame() 
        
        if status == "DISCONNECTED" or status == "UNINITIALIZED":
            # Create a blank, black frame (640x480, 3 channels)
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8) 
            
            # Draw the error message in the center
            message = "KAMERA TERPUTUS" if status == "DISCONNECTED" else "CAMERA ERROR"
            message2 = "PASANG DAN MULAI APLIKASI KEMBALI"
            cv2.putText(error_frame, message, 
                        (150, 240), # Position
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, # Scale
                        (0, 0, 255), # Red color (BGR)
                        3, # Thickness
                        cv2.LINE_AA)
            cv2.putText(error_frame, message2, 
                        (150, 280), # Position
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, # Scale
                        (0, 0, 255), # Red color (BGR)
                        3, # Thickness
                        cv2.LINE_AA)
            
            annotated_frame = error_frame
            time.sleep(0.5) # Reduce loop speed while disconnected
        
        elif not ret or frame is None:
            time.sleep(0.01)
            continue
        
        else:
            # Normal processing only if status is "OK" and frame is valid
            annotated_frame = processor.process_frame(frame)
        
        if annotated_frame is None:
            time.sleep(0.01)
            continue

        _, buffer = cv2.imencode(".jpg", annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

# Root folder
@app.route("/")
def index():
    return render_template("index.html")


# Video Feed
@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# Menghitung data
@app.route("/count_data")
def count_data():
    current_counts = processor.get_counts()

    # Add data to the processor's internal history
    processor.add_graph_data(current_counts)

    return jsonify(current_counts)


@app.route("/reset_count", methods=["POST"])
def reset_count():
    # This also resets the internal graph_data in the processor
    processor.reset_counts()
    return jsonify({"message": "Count reset successful"})


@app.route("/download_excel", methods=["GET"])
def download_excel():
    # Get all necessary data directly from the processor
    historical_data = processor.get_historical_data()

    # NOTE: The processor's graph_data["timestamps"] now holds the full datetime string.
    timestamps = historical_data["timestamps"]
    data_points = historical_data["data_points"]

    wb = Workbook()
    ws = wb.active
    ws.title = "Crowd Data"

    # FIXED: Updated Excel header to include Date and Time
    ws.append(
        ["Tanggal", "Waktu", "Jumlah Pengunjung (Masuk - Keluar)", "Masuk", "Keluar"]
    )

    for timestamp_str, data_point in zip(timestamps, data_points):
        # Assuming processor now stores the full "YYYY-MM-DD HH:MM:SS" string:
        try:
            date_part, time_part = timestamp_str.split(" ")
        except ValueError:
            # Fallback if processor only stores time (HH:MM:SS)
            date_part = datetime.now().strftime(
                "%Y-%m-%d"
            )  # Use current date as a best guess
            time_part = timestamp_str

        ws.append(
            [
                date_part,  # Column 1: Date
                time_part,  # Column 2: Time
                data_point["current"],  # Column 3: Current Count
                data_point["entry"],  # Column 4: Entry Count
                data_point["exit"],  # Column 5: Exit Count
            ]
        )

    excel_buffer = BytesIO()
    wb.save(excel_buffer)
    excel_buffer.seek(0)

    return send_file(
        excel_buffer,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"Crowd_Data_{VERSION.capitalize()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
    )


# Route to GET current configuration (for filling the form)
@app.route('/api/config', methods=['GET'])
def get_frontend_config():
    """Returns the current full config structure to the frontend."""
    # We use the CONFIG global variable loaded at startup
    if CONFIG:
        # We need to restructure the flat form data back into the nested JSON structure
        # for easy display on the frontend.
        frontend_config = {
            "version": CONFIG.get("version", "horizontal"),
            "resize": CONFIG.get("resize", {}),
            "tracking": CONFIG.get("tracking", {}),
            "capacity": CONFIG.get("capacity", {}),
            "horizontal_lines": CONFIG.get("horizontal_lines", {}),
            "vertical_lines": CONFIG.get("vertical_lines", {}),
            "initial_counts": CONFIG.get("initial_counts", {})
        }
        return jsonify(frontend_config)
    return jsonify({"error": "Configuration not loaded"}), 500

# Route to POST new configuration (for saving changes)
@app.route('/api/config', methods=['POST'])
def save_frontend_config():
    global CONFIG # Allow modification of the global CONFIG variable
    data = request.json
    
    if not data:
        return jsonify({"success": False, "message": "No data received."}), 400

    # 1. Start with a copy of the current configuration
    new_config_data = CONFIG.copy() 

    # Update simple and nested keys (non-line related)
    current_version = data.get('version')
    new_config_data["version"] = current_version
    new_config_data["capacity"]["max_crowd_count"] = data.get('max_crowd_count')
    new_config_data["resize"]["width"] = data.get('resize_width')
    new_config_data["tracking"]["max_disappeared"] = data.get('max_disappeared')
    new_config_data["initial_counts"]["entry"] = data.get('initial_entry')
    new_config_data["initial_counts"]["exit"] = data.get('initial_exit')

    # --- NEW LOGIC START ---
    
    # 1a. Store raw line positions from form data (must be converted to int here)
    h_entry_raw = int(data.get('h_entry'))
    h_exit_raw = int(data.get('h_exit'))
    v_entry_raw = int(data.get('v_entry'))
    v_exit_raw = int(data.get('v_exit'))

    # 1b. Handle Swap Direction Flag
    swap_val = data.get('swap_direction')
    is_swapped = bool(int(swap_val)) if isinstance(swap_val, str) and swap_val.isdigit() else bool(swap_val)
    new_config_data["tracking"]["swap_direction"] = is_swapped
    
    
    # 2. Update Horizontal Lines (Gerak Vertikal - Y axis)
    
    # By default, use the raw values from the form:
    h_entry_final = h_entry_raw
    h_exit_final = h_exit_raw
    
    # Check if this version is active AND swap is requested:
    if is_swapped and current_version == "horizontal":
        # If swapped, swap the values for storage
        h_entry_final = h_exit_raw
        h_exit_final = h_entry_raw
        
    new_config_data["horizontal_lines"]["entry_line_position"] = h_entry_final
    new_config_data["horizontal_lines"]["exit_line_position"] = h_exit_final


    # 3. Update Vertical Lines (Gerak Horizontal - X axis)

    # By default, use the raw values from the form:
    v_entry_final = v_entry_raw
    v_exit_final = v_exit_raw
    
    # Check if this version is active AND swap is requested:
    if is_swapped and current_version == "vertical":
        # If swapped, swap the values for storage
        v_entry_final = v_exit_raw
        v_exit_final = v_entry_raw
        
    new_config_data["vertical_lines"]["entry_line_position"] = v_entry_final
    new_config_data["vertical_lines"]["exit_line_position"] = v_exit_final

    # --- NEW LOGIC END ---


    # 4. Save the structure to the file
    if save_config_to_file(new_config_data):
        # Update the global CONFIG dictionary in memory (only applies to future GETs)
        # Note: The active processor is NOT reloaded. A restart is MANDATORY.
        CONFIG = new_config_data 
        
        return jsonify({
            "success": True, 
            "message": "Konfigurasi berhasil disimpan! **APLIKASI HARUS DI-RESTART** untuk menerapkan perubahan."
        })
    else:
        return jsonify({"success": False, "message": "Gagal menyimpan file konfigurasi ke disk."}), 500

@app.route("/petunjuk")
def petunjuk():
    """Menyajikan halaman Petunjuk Penggunaan."""
    return render_template("petunjuk.html")

@app.route('/konfigurasi')
def konfigurasi():
    """Menyajikan halaman Konfigurasi."""
    return render_template('konfigurasi.html')

@app.route("/tentang")
def tentang():
    """Menyajikan halaman Tentang Aplikasi."""
    return render_template("tentang.html")

@app.route("/shutdown", methods=["POST"])
def shutdown():
    """Shuts down the running Flask server."""
    processor.release_resources()
    print("\n--- Menerima sinyal shutdown dari browser. Mematikan aplikasi... ---")
    # Use threading for os._exit to gracefully close the werkzeug server
    threading.Thread(target=lambda: time.sleep(1) or os._exit(0)).start()
    return jsonify({"success": True, "message": "Application is closing."})

@app.route("/recording/toggle", methods=["POST"])
def toggle_recording_route():
    global VERSION  # Make sure the global VERSION variable is accessible

    if processor is None:
        return jsonify({"success": False, "message": "Processor not initialized."}), 503

    action = request.args.get("action")
    if action not in ["start", "stop"]:
        return jsonify({"success": False, "message": "Invalid action parameter."}), 400

    try:
        # --- FIX: Pass the VERSION variable to the processor method ---
        success, result_data = processor.toggle_recording(action, VERSION)

        if success:
            file_path = result_data if action == "stop" else None
            return jsonify({"success": True, "action": action, "file_path": file_path})
        else:
            return jsonify({"success": False, "message": result_data})

    except IOError as e:
        return jsonify({"success": False, "message": f"IO Error: {e}"}), 500
    except Exception as e:
        return jsonify({"success": False, "message": f"Unknown error: {e}"}), 500


@app.route('/recording/status', methods=['GET'])
def get_recording_status_route():
    if processor is None:
        return jsonify({"is_recording": False}), 503
    
    status = processor.get_recording_status()
    return jsonify({"is_recording": status})

# --- UTILITY AND RUN APP ---
def open_browser():
    # Wait for the Flask server to start up
    time.sleep(2)

    app_url = "http://127.0.0.1:5000/"

    browser_paths = [
        "C:/Program Files/Google/Chrome/Application/chrome.exe",
        "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    ]

    launched = False

    for browser_path in browser_paths:
        if os.path.exists(browser_path):
            try:
                # Use subprocess to run the command with the --app flag
                subprocess.Popen(
                    [browser_path, f"--app={app_url}", "--start-fullscreen"]
                )
                print(
                    f"Launching app in App Mode using: {os.path.basename(browser_path)}"
                )
                launched = True
                break
            except Exception as e:
                print(f"Error launching {os.path.basename(browser_path)}: {e}")
                continue

    if not launched:
        print("No App Mode browser found. Falling back to default browser tab.")
        webbrowser.open_new_tab(app_url)


if __name__ == "__main__":
    # Call the browser launcher directly (blocking but safe in this context)
    try:
        open_browser()
    except Exception as e:
        print(f"Warning: Failed to launch browser automatically: {e}")
        print("Access the application manually at http://127.0.0.1:5000/")

    # Run the main Flask application
    app.run(debug=True, use_reloader=False)