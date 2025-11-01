# MOV COUNTER VERTICAL. 
# Hijau sisi masuk pengunjung
# Merah sisi keluar pengunjung

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
from threading import Thread
import threading

print("######################################")
print("# Memulai Aplikasi Crowded Detection #")
print("######################################")
print("Versi: Vertical Line")
print("Tunggu...")

app = Flask(__name__, static_folder='static')

# Inisialisasi YOLOv8 model
print("YOLOv8: Loading Computer Vision Model")
try:
    model = YOLO('head.pt') 
except Exception as e:
    print(f"Error loading Yolo model: {e}")
    model = None
print("Status: Model loaded successfully.")

# Inisialisasi VideoCapture
print("OpenCV: Inisialisasi Webcam...")
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Tidak bisa membuka KAMERA")
except IOError as e:
    print(f"FATAL ERROR: {e}. Kamera mungkin tidak terpasang.")
    cap = None
print("Status: Inisialisasi berhasil. Memulai Flask server...")

# KODINGAN ADIB
entry_line_position = 440
exit_line_position = 200
entry_count = 150 # Dimulai dari 0
exit_count = 100 # Dimulai dari 0
resize_width = 640   # Ubah sesuai kebutuhan
resize_height = 480  # Ubah sesuai kebutuhan

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
object_prev_positions = {}
counted_on_entry = set()
counted_on_exit = set()

def detect_direction(object_id, cx, prev_cx):
    global entry_count, exit_count
    if prev_cx is not None:
        if prev_cx < exit_line_position <= cx:  # bergerak melewati exit line (ke kanan)
            if object_id not in counted_on_exit:
                exit_count += 1
                counted_on_exit.add(object_id)
                counted_on_entry.discard(object_id)
            return "Exit"
        elif prev_cx > entry_line_position >= cx:  # bergerak melewati entry line (ke kiri)
            if object_id not in counted_on_entry:
                entry_count += 1
                counted_on_entry.add(object_id)
                counted_on_exit.discard(object_id)
            return "Entry"
    else:
        if object_id in counted_on_entry or object_id in counted_on_exit:
            return "Idle"
    return "Orang"

def generate_frames():
    global entry_count, exit_count

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

        for object_id, centroid in objects.items():
            cx, cy = centroid
            prev_cx = object_prev_positions.get(object_id)
            direction = detect_direction(object_id, cx, prev_cx)
            object_prev_positions[object_id] = cx

            # Draw bounding box and direction label on the frame
            for (x1, y1, x2, y2) in rects:
                if (x1 + x2) // 2 == cx and (y1 + y2) // 2 == cy:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, direction, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        dalam_count = entry_count - exit_count

        cv2.line(frame, (entry_line_position, 0), (entry_line_position, frame.shape[0]), (0, 100, 0), 2)  # Entry line (green)
        cv2.line(frame, (exit_line_position, 0), (exit_line_position, frame.shape[0]), (0, 0, 255), 2)    # Exit line (red)

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
    global entry_count, exit_count, graph_data
    current_count = entry_count - exit_count

    # Tambahkan data waktu dan jumlah
    graph_data["time_labels"].append(datetime.now().strftime("%H:%M:%S"))
    graph_data["count_data"].append(current_count)

    data = {
        "entry_count": entry_count,
        "exit_count": exit_count,
        "current_count": current_count
    }
    return jsonify(data)

@app.route('/reset_count', methods=['POST'])
def reset_count():
    global entry_count, exit_count, current_count
    entry_count = 0
    exit_count = 0
    current_count = 0
    return jsonify({"message": "Count reset successful"})


@app.route('/download_excel', methods=['GET'])
def download_excel():
    # Buat workbook Excel
    wb = Workbook()
    ws = wb.active
    ws.title = "Crowd Data"

    # Tambahkan data ke worksheet
    ws.append(["Time", "Current Count", "Entry Count", "Exit Count"])
    for time, count in zip(graph_data["time_labels"], graph_data["count_data"]):
        ws.append([time, count, entry_count, exit_count])

    # Buat grafik menggunakan matplotlib
    plt.figure(figsize=(10, 6))
    plt.plot(graph_data["time_labels"], graph_data["count_data"], marker='o', linestyle='-', color='blue')
    plt.title("Visitor Count Over Time")
    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()

    # Simpan grafik ke gambar
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    plt.close()
    img_buffer.seek(0)

    # Masukkan gambar grafik ke worksheet
    img = Image(img_buffer)
    ws.add_image(img, 'E2')  # Letakkan grafik di sel E2

    # Simpan workbook ke buffer
    excel_buffer = BytesIO()
    wb.save(excel_buffer)
    excel_buffer.seek(0)

    # Kirim file ke pengguna
    return send_file(
        excel_buffer,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="Crowd_Data.xlsx"
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

