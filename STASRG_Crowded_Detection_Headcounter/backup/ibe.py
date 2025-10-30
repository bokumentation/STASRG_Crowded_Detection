import cv2
import numpy as np
import pandas as pd
import webbrowser
import threading
import time
from ultralytics import YOLO
from scipy.spatial import distance as dist
from collections import OrderedDict, deque
from flask import Flask, Response, render_template, jsonify, send_file, request
from datetime import datetime
from io import BytesIO
from threading import Thread

# --- THREAD-SAFE SHARED STATE CLASS ---
class SharedState:
    """Holds the latest CV results and provides thread safety via Lock."""
    def __init__(self, max_data_len=500):
        self.lock = threading.Lock()
        self.current_frame = None  # Latest processed JPEG frame (bytes)
        self.current_count = 0
        self.total_count = 0
        self.visitor_data = deque(maxlen=max_data_len) # Data for excel download

    def update_frame(self, frame_bytes, current_count, total_count):
        """Called by the CV Thread to update the latest frame and counts."""
        with self.lock:
            self.current_frame = frame_bytes
            self.current_count = current_count
            self.total_count = total_count

    def get_data(self):
        """Called by Flask routes to safely get the current data and frame."""
        with self.lock:
            return (self.current_frame, 
                    self.current_count, 
                    self.total_count, 
                    list(self.visitor_data))
    
    def clear_visitor_data(self):
        """Safely clears the visitor data log."""
        with self.lock:
            self.visitor_data.clear()

# --- GLOBAL STATE & INITIALIZATION ---
app = Flask(__name__)
shared_state = SharedState()
reset_flag = False

# Inisialisasi YOLOv8 model
try:
    model = YOLO('survei2.pt') 
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

# Inisialisasi Video Capture
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
except IOError as e:
    print(f"FATAL ERROR: {e}. Camera may not be available.")
    cap = None

confidence_threshold = 0.3


# --- CENTROID TRACKER CLASS ---
class CentroidTracker:
    def __init__(self, max_disappeared=50, distance_threshold=50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.distance_threshold = distance_threshold

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
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            if not object_centroids or not input_centroids.any():
                return self.objects
            
            D = dist.cdist(np.array(object_centroids), input_centroids)

            if D.shape[0] == 0 or D.shape[1] == 0:
                return self.objects
        
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set() 

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > self.distance_threshold:
                    continue
            
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects


# --- CV WORKER FUNCTION (THE PRODUCER) ---
def cv_worker(shared_state):
    """
    Runs in a dedicated thread. Continuously processes frames, updates the tracker, 
    and writes the latest results to the shared_state object.
    """
    global reset_flag

    if cap is None or model is None:
        print("CV Worker failed to start: Camera or model not initialized.")
        return

    # Local state for the worker thread
    worker_dt = CentroidTracker()
    worker_total_count = 0
    worker_last_saved_time = datetime.now()

    while True:
        # 1. Handle Reset Flag
        if reset_flag:
            worker_dt = CentroidTracker() # Re-initialize tracker
            worker_total_count = 0
            shared_state.clear_visitor_data()
            reset_flag = False
            # Update shared state immediately after reset
            shared_state.update_frame(None, 0, 0)
        
        # 2. Read Frame
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        # 3. YOLO Detection
        results = model(frame, verbose=False)
        persons = [d for d in results[0].boxes.data.tolist() if int(d[5]) == 0 and d[4] >= confidence_threshold]
        rects = []
    
        # 4. Draw Bounding Boxes and Prepare Rectangles
        for x1, y1, x2, y2, conf, cls in persons:
            rects.append((int(x1), int(y1), int(x2), int(y2)))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            # Conf text omitted for simplicity, but can be added here
            
        # 5. Update Tracker
        objects = worker_dt.update(rects)
        worker_current_count = len(objects)

        if worker_dt.next_object_id > worker_total_count:
            worker_total_count = worker_dt.next_object_id

        # 6. Draw IDs and Centroids
        for objectID, centroid in objects.items():
            text = f"ID {objectID}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # 7. Display Counts on Frame
        # count_text = f"Current: {worker_current_count} | Total: {worker_total_count}"
        # cv2.putText(frame, count_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 8. Periodic Data Saving (1 minute interval)
        timestamp = datetime.now()
        if (timestamp - worker_last_saved_time).total_seconds() >= 60:
            # Safely append data to the shared state's deque
            with shared_state.lock:
                shared_state.visitor_data.append({"time": timestamp.strftime('%H:%M:%S'), 
                                                  "count": worker_current_count})
            worker_last_saved_time = timestamp

        # 9. Convert frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # 10. Update the shared state (Thread-safe write)
        shared_state.update_frame(frame_bytes, worker_current_count, worker_total_count)
        
        # Optional: Sleep to prevent the worker from consuming excessive CPU if processing is extremely fast
        # time.sleep(0.01)


# --- FLASK ROUTES (THE CONSUMERS) ---

@app.route('/')
def index():
    return render_template('hasilcount.html')

@app.route('/video_feed')
def video_feed():
    def streaming_generator():
        while True:
            # Read the latest frame from the shared state (Thread-safe read)
            frame_bytes, _, _, _ = shared_state.get_data()
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                 # If no frame is available (e.g., camera error or reset), pause
                time.sleep(0.1) 
            time.sleep(0.03) # Control streaming rate (approx 30 FPS)
            
    if cap is None:
        return Response("Error: Camera not initialized.", status=503)
    return Response(streaming_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/count_data')
def count_data():
    _, current_count, total_count, _ = shared_state.get_data()
    data = {
        "current_count": current_count,
        "total_count": total_count
    }
    return jsonify(data)

@app.route('/visitor_data')
def visitor_data_route():
    _, _, _, visitor_data_list = shared_state.get_data()
    return jsonify(visitor_data_list)

@app.route('/download_excel')
def download_excel():
    _, _, _, visitor_data_list = shared_state.get_data()
    
    # Buat data frame dari data pengunjung
    df = pd.DataFrame(visitor_data_list)

    # Simpan file excel ke memori menggunakan BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Visitor Data')

    # Kirim file untuk diunduh
    output.seek(0)
    return send_file(output, 
                    as_attachment=True, 
                    download_name='visitor_data.xlsx', 
                    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

@app.route('/currentvisitor')
def currentvisitor():
    return render_template('currentvisitor.html')

@app.route('/crowd_status')
def crowd_status():
    _, current_count, _, _ = shared_state.get_data()
    status = "Crowd" if current_count <= 200 else "Overcrowded"
    return jsonify({'status': status})

@app.route('/reset_count', methods=['POST'])
def reset_count():
    global reset_flag
    reset_flag = True # Signal the worker thread to perform the reset
    return jsonify({"message": "Count reset scheduled", 
                    "current_count": 0, 
                    "total_count": 0})

# --- UTILITY AND RUN APP ---
def open_browser():
    time.sleep(1) # Give the server a moment to start
    webbrowser.open_new_tab('http://127.0.0.1:5000/')

if __name__ == "__main__":

    if cap is not None and model is not None:
        
        # 1. Initialize and Start the CV Worker Thread (The Producer)
        # This thread runs the high-load CV processing in the background.
        cv_thread = threading.Thread(target=cv_worker, args=(shared_state,))
        cv_thread.daemon = True  # Ensures the thread shuts down when the main program exits
        cv_thread.start()
        print("Background CV Worker Thread started.")

        # 2. Launch browser
        Thread(target=open_browser).start()

    # 3. Start the Flask App (The Consumer)
    app.run(debug=True, use_reloader=False)