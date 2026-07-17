from flask import Flask, Response, render_template, jsonify, send_file
import cv2
import numpy as np
from ultralytics import YOLO
from collections import OrderedDict
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from io import BytesIO
import matplotlib.pyplot as plt
from datetime import datetime

app = Flask(__name__, static_folder='static')

# YOLO setup
model = YOLO('head.pt')  # Path to your YOLO model
entry_line_position = 320
exit_line_position = 160
entry_count = 150
exit_count = 90
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

# Video capture
# videopath = "IMG_20241122_114318957_BURST0005.jpg"
cap = cv2.VideoCapture(0)

def detect_direction(object_id, cy, prev_cy):
    global entry_count, exit_count
    if prev_cy is not None:
        if prev_cy < exit_line_position <= cy:  # Bergerak melewati exit line (ke bawah)
            if object_id not in counted_on_exit:  # Hanya hitung jika belum dihitung sebelumnya
                exit_count += 1
                counted_on_exit.add(object_id)  # Tandai objek sebagai dihitung di exit
                if object_id in counted_on_entry:  # Hapus dari entry jika berpindah garis
                    counted_on_entry.remove(object_id)
            return "Exit"
        elif prev_cy > entry_line_position >= cy:  # Bergerak melewati entry line (ke atas)
            if object_id not in counted_on_entry:  # Hanya hitung jika belum dihitung sebelumnya
                entry_count += 1
                counted_on_entry.add(object_id)  # Tandai objek sebagai dihitung di entry
                if object_id in counted_on_exit:  # Hapus dari exit jika berpindah garis
                    counted_on_exit.remove(object_id)
            return "Entry"
    else:
        # Jika objek tidak bergerak, jangan hitung
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
        results = model(frame)
        rects = []

        for result in results:
            for i, bbox in enumerate(result.boxes.xyxy):  # Get the bounding box
                x1, y1, x2, y2 = map(int, bbox[:4])  # Coordinates of bounding box
                rects.append((x1, y1, x2, y2))  # Add bounding box to list

        objects = ct.update(rects)

        for object_id, centroid in objects.items():
            cx, cy = centroid
            prev_cy = object_prev_positions.get(object_id)
            direction = detect_direction(object_id, cy, prev_cy)
            object_prev_positions[object_id] = cy

            # Draw bounding box and direction label on the frame
            for (x1, y1, x2, y2) in rects:
                if (x1 + x2) // 2 == cx and (y1 + y2) // 2 == cy:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, direction, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        dalam_count = entry_count - exit_count

        cv2.line(frame, (0, entry_line_position), (frame.shape[1], entry_line_position), (0, 100, 0), 2)  # Entry line (green)
        cv2.line(frame, (0, exit_line_position), (frame.shape[1], exit_line_position), (0, 0, 255), 2)    # Exit line (red)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

graph_data = {
    "time_labels": [],
    "count_data": []
}

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

if __name__ == "__main__":
    app.run(debug=True)

