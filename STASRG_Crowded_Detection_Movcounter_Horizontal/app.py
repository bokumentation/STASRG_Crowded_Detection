# MOV COUNTER HORIZONTAL. 

import webbrowser
import subprocess
import time
import os
from flask import Flask, Response, render_template, jsonify, send_file, request
import cv2
import numpy as np
from ultralytics import YOLO
from collections import OrderedDict
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from io import BytesIO
import matplotlib.pyplot as plt
from datetime import datetime
from threading import Thread, Lock
import threading

print("######################################")
print("# Memulai Aplikasi Crowded Detection #")
print("######################################")
print("Versi: Horizontal Line | Final")
print("Tunggu...")

app = Flask(__name__, static_folder='static')

# Inisialisasi YOLOv8 model
print("YOLOv8: Loading Computer Vision Model. Tunggu...")
try:
    model = YOLO('head.pt') 
except Exception as e:
    print(f"Error loading Yolo model: {e}")
    model = None
print("Status: Model loaded successfully.")

# Inisialisasi VideoCapture
print("OpenCV: Inisialisasi OpenCV dan Kamera...")
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Tidak bisa membuka KAMERA")
except IOError as e:
    print(f"FATAL ERROR: {e}. Kamera mungkin tidak terpasang.")
    cap = None
print("Status: Inisialisasi berhasil. Memulai Flask server...")

data_lock = Lock()

# KODINGAN ADIB
# Horizontal Line
entry_line_position = 320
exit_line_position = 160
entry_count = 0 # Dimulai dari 0
exit_count = 0 # Dimulai dari 0
resize_width = 640   # Ubah sesuai kebutuhan
resize_height = 480  # Ubah sesuai kebutuhan

last_log_time = datetime.now()
log_data = []
object_prev_positions = {}
counted_on_entry = set()
counted_on_exit = set()
log_interval_seconds = 5 # Log data every 5 seconds

# Centroid tracker class to assign unique IDs to objects and track them
class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Assign centroids to existing objects based on proximity
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])) - used_rows
            unused_cols = set(range(0, D.shape[1])) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects

ct = CentroidTracker()

# cy for the horizontal
def detect_direction(object_id, cy, prev_cy):
    global entry_count, exit_count, counted_on_entry, counted_on_exit
    if prev_cy is not None:
        if prev_cy < cy:
            if prev_cy < entry_line_position <= cy:  # Bergerak melewati exit line (ke bawah)
                if object_id not in counted_on_exit:  # Hanya hitung jika belum dihitung sebelumnya
                    exit_count += 1
                    counted_on_exit.add(object_id)  # Tandai objek sebagai dihitung di exit
                    if object_id in counted_on_entry:  # Hapus dari entry jika berpindah garis
                        counted_on_entry.remove(object_id)
            return "Exit"
        elif prev_cy > cy:
            if prev_cy > exit_line_position >= cy:  # Bergerak melewati entry line (ke atas)
                if object_id not in counted_on_entry:  # Hanya hitung jika belum dihitung sebelumnya
                    entry_count += 1
                    counted_on_entry.add(object_id)  # Tandai objek sebagai dihitung di entry
                    if object_id in counted_on_exit:  # Hapus dari exit jika berpindah garis
                        counted_on_exit.remove(object_id)
            return "Entry"
        
    if prev_cy is not None and abs(cy - entry_line_position) > 100 and object_id in counted_on_entry:
        counted_on_entry.discard(object_id)
    if prev_cy is not None and abs(cy - exit_line_position) > 100 and object_id in counted_on_exit:
        counted_on_exit.discard(object_id)

    # Check if object is far below the lower line (Entry=320)
    if cy > (entry_line_position + 50) and object_id in counted_on_exit:
        counted_on_exit.discard(object_id)
        
    # Check if object is far above the upper line (Exit=160)
    if cy < (exit_line_position - 50) and object_id in counted_on_entry:
        counted_on_entry.discard(object_id)

    # 3. Tracking Status Reporting
    if object_id in counted_on_entry or object_id in counted_on_exit:
        return "Counted"
    
    return "Idle"

# Horizontal
def generate_frames():
    global entry_count, exit_count, last_log_time

    # Check for offline status (added a block for completeness, though omitted in your upload)
    if cap is None or model is None:
        blank_frame = np.zeros((resize_height, resize_width, 3), dtype=np.uint8)
        cv2.putText(blank_frame, "OFFLINE", (50, resize_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', blank_frame)
        frame = buffer.tobytes()
        while True:
            time.sleep(1)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\r\n')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (resize_width, resize_height))
        results = model(frame, verbose=False) # ibe added verbose false
        rects = []

        for result in results:
            for i, bbox in enumerate(result.boxes.xyxy):  # Get the bounding box
                x1, y1, x2, y2 = map(int, bbox[:4])  # Coordinates of bounding box
                rects.append((x1, y1, x2, y2))  # Add bounding box to list

        objects = ct.update(rects)

        with data_lock:
            frame_directions = {}

            for object_id, centroid in objects.items():
                cx, cy = centroid
                prev_cy = object_prev_positions.get(object_id)
                direction = detect_direction(object_id, cy, prev_cy)
                object_prev_positions[object_id] = cy
                frame_directions[object_id] = direction

            # # Local copies of counts for drawing and logging (READ)
            # current_entry_count = entry_count
            # current_exit_count = exit_count

            # Local copies of counts for drawing and logging (READ)
            current_entry_count = entry_count
            current_exit_count = exit_count
            current_dalam_count = current_entry_count - current_exit_count

            # --- LOGGING (WRITE Shared State) ---
            current_time = datetime.now()
            if (current_time - last_log_time).total_seconds() >= log_interval_seconds:
                log_data.append({
                    "time": current_time.strftime("%H:%M:%S"),
                    "entry": current_entry_count,
                    "exit": current_exit_count,
                    "current": current_dalam_count
                })
                # Update time while inside the lock
                last_log_time = current_time
        
        # Log release


        current_dalam_count = current_entry_count - current_exit_count
        
        # draw bounding boxes
        rect_index = 0
        for object_id, centroid in objects.items():
            if rect_index < len(rects): # Safety check
                x1, y1, x2, y2 = rects[rect_index]
                direction = frame_directions.get(object_id, "Unknown")
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{object_id}:{direction}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                rect_index += 1

        # Draw line
        cv2.line(frame, (0, entry_line_position), (frame.shape[1], entry_line_position), (0, 100, 0), 2)
        cv2.line(frame, (0, exit_line_position), (frame.shape[1], exit_line_position), (0, 0, 255), 2)

        cv2.putText(frame, f"IN: {current_entry_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"OUT: {current_exit_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"CURRENT: {current_dalam_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)

            # # Draw bounding box and direction label on the frame
            # for (x1, y1, x2, y2) in rects:
            #     if (x1 + x2) // 2 == cx and (y1 + y2) // 2 == cy:
            #         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            #         cv2.putText(frame, direction, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # # dalam_count = entry_count - exit_count
 

        # for object_id, centroid in objects.items():
        #     cx, cy = centroid
        #     # Note: Redetermining direction here is safe for display label purposes 
        #     # as it only reads state (or uses the direction value obtained inside the lock).
        #     direction = detect_direction(object_id, cy, object_prev_positions.get(object_id)) 
            
        #     # Draw bounding box and direction label on the frame
        #     for (x1, y1, x2, y2) in rects:
        #         if (x1 + x2) // 2 == cx and (y1 + y2) // 2 == cy:
        #             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        #             cv2.putText(frame, direction, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        #             break

        # cv2.line(frame, (0, entry_line_position), (frame.shape[1], entry_line_position), (0, 100, 0), 2)  # Entry line (green)
        # cv2.line(frame, (0, exit_line_position), (frame.shape[1], exit_line_position), (0, 0, 255), 2)    # Exit line (red)

        # LOGGING (WRITE Shared State)
        current_time = datetime.now()
        if (current_time - last_log_time).total_seconds() >= log_interval_seconds:
            with data_lock:
                log_data.append({
                    "time": current_time.strftime("%H:%M:%S"),
                    "entry": current_entry_count,
                    "exit": current_exit_count,
                    "current": current_dalam_count
                })
            last_log_time = current_time


        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Root folder
@app.route('/')
def index():
    return render_template('index.html')

# Video Feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

graph_data = {
    "time_labels": [],
    "count_data": []
}



# Menghitung data
@app.route('/count_data')
def count_data():
    global entry_count, exit_count

    with data_lock:
        entry = entry_count
        exit = exit_count

    current_count = entry - exit

    timestamp = datetime.now().strftime("%H:%M:%S")

    # log_data.append({
    #     "time": timestamp,
    #     "entry": entry_count,
    #     "exit": exit_count,
    #     "current": current_count
    # })

    data = {
        "entry_count": entry_count,
        "exit_count": exit_count,
        "current_count": current_count
    }
    return jsonify(data)

@app.route('/reset_count', methods=['POST'])
def reset_count():
    # global entry_count, exit_count, current_count
    global entry_count, exit_count, ct, object_prev_positions, counted_on_entry, counted_on_exit

    with data_lock:
        # Reset Counters
        entry_count = 0
        exit_count = 0
        
        # Reset Tracker State
        global ct
        ct = CentroidTracker()
        object_prev_positions = {}
        counted_on_entry = set()
        counted_on_exit = set()

    # entry_count = 0
    # exit_count = 0
    # current_count = 0
    return jsonify({"message": "Count reset successful"})


@app.route('/download_excel', methods=['GET'])
def download_excel():
    # Buat workbook Excel
    global log_data

    with data_lock:
        local_log_data = list(log_data)

    wb = Workbook()
    ws = wb.active
    ws.title = "Crowd Data"

    # Tambahkan Header ke worksheet
    ws.append(["Time", "Current Count", "Entry Count", "Exit Count"])
    
    for item in local_log_data:
        ws.append([
            item["time"], 
            item["current"], 
            item["entry"], 
            item["exit"]
        ])

    # Simpan workbook ke buffer
    excel_buffer = BytesIO()
    wb.save(excel_buffer)
    excel_buffer.seek(0)

    # Kirim file ke pengguna
    return send_file(
        excel_buffer,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"Crowd_Data_MovCount_Horizontal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx" 
    )

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Shuts down the running Flask server."""
    func = request.environ.get('werkzeug.server.shutdown')
    
    if cap and cap.isOpened():
        cap.release() 

    print("\n--- Menerima sinyal shutdown dari browser. Mematikan aplikasi... ---")
    threading.Thread(target=lambda: time.sleep(1) or os._exit(0)).start()
    return jsonify({"success": True, "message": "Application is closing."})

# --- UTILITY AND RUN APP ---
def open_browser():
    """Launches the application URL in a browser's 'App Mode' for a PWA-like 
    experience, falling back from Chrome to Edge if Chrome is not found.
    """
    # Wait for the Flask server and the CV thread to start up
    time.sleep(2) 
    
    app_url = 'http://127.0.0.1:5000/'
    
    browser_paths = [
        "C:/Program Files/Google/Chrome/Application/chrome.exe",
        "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    ]

    launched = False
    
    for browser_path in browser_paths:
        if os.path.exists(browser_path):
            try:
                # Use subprocess to run the command with the --app flag
                subprocess.Popen([
                    browser_path, 
                    f"--app={app_url}",
                    "--start-fullscreen"
                    ])
                print(f"Launching app in App Mode using: {os.path.basename(browser_path)}")
                launched = True
                break # Exit the loop once successfully launched
                
            except Exception as e:
                print(f"Error launching {os.path.basename(browser_path)}: {e}")
                continue # Try the next browser path

    if not launched:
        # Fallback to standard webbrowser if no App Mode browser could be found
        print("No App Mode browser found. Falling back to default browser tab.")
        webbrowser.open_new_tab(app_url)

if __name__ == "__main__":
    Thread(target=open_browser).start()
    app.run(debug=False, use_reloader=False)

