import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from collections import OrderedDict
import json
from cv_processor_vertical import CentroidTracker
from datetime import datetime
from path_resolver import get_resource_path

# --- HORIZONTAL PROCESSOR IMPLEMENTATION ---
class HorizontalProcessor:
    """Handles YOLO, Video, Tracking, and Counting for Horizontal Lines."""

    def __init__(self, config):
        self.config = config

        # Read initial values from config.json
        self.entry_line_position = self.config["horizontal_lines"][
            "entry_line_position"
        ]
        self.exit_line_position = self.config["horizontal_lines"]["exit_line_position"]
        initial_entry = self.config["initial_counts"].get("entry", 0)
        initial_exit = self.config["initial_counts"].get("exit", 0)
        self.entry_count = initial_entry
        self.exit_count = initial_exit

        self.ct = CentroidTracker(
            max_disappeared=self.config["tracking"]["max_disappeared"]
        )
        self.object_prev_positions = {}
        self.counted_on_entry = set()
        self.counted_on_exit = set()

        self.model = self._load_model()
        self.cap = self._initialize_camera()
        self.graph_data = {"timestamps": [], "data_points": []}  # Historical data

        # Recording setup
        self.is_recording = False
        self.video_writer = None
        self.frame_width = self.config["resize"]["width"]
        self.frame_height = self.config["resize"]["height"]
        self.last_recording_path = None

        # 1. Determine the Base Folder (~/Videos or ~/My Videos)
        # We try to find the standard 'Videos' folder within the user's home directory (~).
        user_home = str(Path.home()) 
        
        # Check common names: 'Videos', 'My Videos' (common on older Windows)
        videos_path = Path(user_home) / "Videos"
        if not videos_path.exists():
            videos_path = Path(user_home) / "My Videos"
            
        # 2. Determine the final project subdirectory name from config
        project_sub_dir = self.config["recording"]["output_directory"] 
        
        # 3. Combine them to create the final recording directory
        final_output_dir = videos_path / project_sub_dir
        
        # Convert to string and ensure the directory exists
        self.recording_dir = str(final_output_dir)
        os.makedirs(self.recording_dir, exist_ok=True)
        
        print(f"Status: Lokasi output video -> {self.recording_dir}")

    def toggle_recording(self, action, version_type):
        if action == "start" and not self.is_recording:
            filename = os.path.join(
                self.recording_dir,
                f"Recording_{version_type.capitalize()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
            )

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                filename, fourcc, 15.0, (self.frame_width, self.frame_height)
            )

            if not self.video_writer.isOpened():
                raise IOError("Could not initialize video writer.")

            self.last_recording_path = filename

            self.is_recording = True
            print(f"Status: Recording started -> {filename}")
            return True, filename

        elif action == "stop" and self.is_recording:
            file_path = self.last_recording_path

            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False
            print(f"Status: Recording stopped -> {file_path}")
            return True, file_path  # Return the stored path

        return False, "Already in the requested state."

    def _load_model(self):
        model_file_name = self.config["model_path"]

        # Use resolver to find the absolute path of the model file
        absolute_model_path = get_resource_path(model_file_name)
        
        print(f"YOLOv8: Memuat Model untuk versi Horizontal dari {absolute_model_path}...")
        try:
            model = YOLO(absolute_model_path)
            print("YOLOV8: Status: Model berhasil dimuat.")
            return model
        except Exception as e:
            print(f"Error loading Yolo model: {e}")
            return None

    def _initialize_camera(self):
        video_source = self.config["video_source"]
        print(f"OpenCV: Inisialisasi Sumber Video ({video_source})...")
        print("Tunggu...")
        try:
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                raise IOError(f"Tidak bisa membuka Sumber Video: {video_source}")
            print("Status: Inisialisasi berhasil.")
            return cap
        except IOError as e:
            print(f"FATAL ERROR: {e}.")
            cap = None

    
    def get_recording_status(self):
        """Returns the current recording status."""
        return self.is_recording

    def get_frame(self):
        if self.cap is None:
            return None, False, "UNINITIALIZED"

        ret, frame = self.cap.read()
        
        if not ret:
            # Check if running a video file and it has ended
            if isinstance(self.config["video_source"], str):
                # Video file ended: loop it
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if ret:
                    return frame, ret, "OK"
                
            # If ret is False AND it's not a video loop reset, assume camera failure
            print("WARNING: GAGAL MEMBUKA KAMERA (ret=False). ASUMSI KONEKSI KE KAMERA TERPUTUS.")
            return None, False, "DISCONNECTED"
            
        return frame, ret, "OK" # Success case

    def release_resources(self):
        if self.cap:
            self.cap.release()
            print("Status: Video source released.")

        if self.video_writer is not None:
            self.video_writer.release()
            print("Status: Video writer released.")

    # --- DIRECTION LOGIC (DIFFERENT) ---
    def _detect_direction(self, object_id, cy, prev_cy):
        """Detects and updates entry/exit counts based on object's vertical (y) movement."""
        # Removed: with self.data_lock:
        if prev_cy is not None:
            # Entry logic (Upward movement)
            if prev_cy > self.entry_line_position >= cy:
                if object_id not in self.counted_on_entry:
                    self.entry_count += 1
                    self.counted_on_entry.add(object_id)
                    self.counted_on_exit.discard(object_id)
                return "Entry"

            # Exit logic (Downward movement)
            elif prev_cy < self.exit_line_position <= cy:
                if object_id not in self.counted_on_exit:
                    self.exit_count += 1
                    self.counted_on_exit.add(object_id)
                    self.counted_on_entry.discard(object_id)
                return "Exit"

        return "Orang"

    def process_frame(self, frame):
        if frame is None or self.model is None:
            return None

        width = self.config["resize"]["width"]
        height = self.config["resize"]["height"]
        frame = cv2.resize(frame, (width, height))

        results = self.model(frame, verbose=False)
        rects = []
        for result in results:
            for bbox in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, bbox[:4])
                rects.append((x1, y1, x2, y2))

        objects = self.ct.update(rects)

        for object_id, centroid in objects.items():
            cx, cy = centroid
            prev_cy = self.object_prev_positions.get(object_id)
            direction = self._detect_direction(object_id, cy, prev_cy)
            self.object_prev_positions[object_id] = cy

            # OLD Bounding Box Drawing Logic (in both files)
            # for (x1, y1, x2, y2) in rects:
            #     if (x1 + x2) // 2 == cx and (y1 + y2) // 2 == cy:
            #         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
            #         cv2.putText(frame, direction, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            #         break

            # MODIFIED Bounding Box Drawing Logic (in both files)
            for x1, y1, x2, y2 in rects:
                if (x1 + x2) // 2 == cx and (y1 + y2) // 2 == cy:

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

                    label = f"ID {object_id} - {direction}"
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        2,
                    )
                    break

        # --- Inside cv_processor_horizontal.py: ---
        entry_color = (0, 200, 0)  # Lime Green (BGR)
        exit_color = (0, 0, 255)  # Orange-Red (BGR)

        # 2. Create a transparent overlay layer
        overlay = frame.copy()
        alpha = 0.5  # Transparency factor (50%)

        # 3. Draw Lines onto the overlay
        # Entry line (Horizontal)
        cv2.line(
            overlay,
            (0, self.entry_line_position),
            (frame.shape[1], self.entry_line_position),
            entry_color,
            2,
        )
        # Exit line (Horizontal)
        cv2.line(
            overlay,
            (0, self.exit_line_position),
            (frame.shape[1], self.exit_line_position),
            exit_color,
            2,
        )

        # 4. Blend the overlay with the original frame
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # NEW: Added log current counts
        # 1. Get current counts (safe access within the processor)
        current_counts = self.get_counts()

        entry_text = f"ENTRY: {current_counts['entry_count']}"
        exit_text = f"EXIT: {current_counts['exit_count']}"
        current_text = f"CURRENT: {current_counts['current_count']}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        text_color = (255, 255, 255)  # White text
        bg_color = (0, 0, 0)  # Black background (Optional: makes text clearer)

        # 2. Draw Entry Count (Top Left)
        cv2.putText(
            frame, entry_text, (10, 30), font, scale, text_color, thickness, cv2.LINE_AA
        )
        # 3. Draw Exit Count (Below Entry)
        cv2.putText(
            frame, exit_text, (10, 60), font, scale, text_color, thickness, cv2.LINE_AA
        )
        # 4. Draw Current Count (Below Exit - Highlighted)
        cv2.putText(
            frame,
            current_text,
            (10, 90),
            font,
            scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )  # Larger, Red for emphasis

        # --- NEW: TIMESTAMP OVERLAY ---
        # Get the current time formatted as a string (e.g., 2025-11-11 10:55:29)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Define text properties (Top Right Corner)
        ts_font = cv2.FONT_HERSHEY_SIMPLEX
        ts_scale = 0.5
        ts_thickness = 1
        ts_color = (250, 250, 250)  # Bright Red color for high visibility (BGR format)

        # Calculate position for Top Right corner placement
        # Get the size of the text to offset placement correctly
        (text_width, _), _ = cv2.getTextSize(timestamp, ts_font, ts_scale, ts_thickness)
        # Place the text slightly inset from the top and right edges
        ts_x = (
            frame.shape[1] - text_width - 10
        )  # Frame width - text width - 10 pixels margin
        ts_y = 30  # 30 pixels down from the top edge

        # Draw the timestamp onto the frame
        cv2.putText(
            frame,
            timestamp,
            (ts_x, ts_y),
            ts_font,
            ts_scale,
            ts_color,
            ts_thickness,
            cv2.LINE_AA,
        )

        # --- END TIMESTAMP OVERLAY ---

        # NEW: Write the annotated frame to disk if recording is active
        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(frame)

        return frame

    def get_counts(self):
        """Returns the current entry and exit counts."""
        # Removed: with self.data_lock:
        return {
            "entry_count": self.entry_count,
            "exit_count": self.exit_count,
            "current_count": self.entry_count - self.exit_count,
        }

    def reset_counts(self):
        """Resets all counting and tracking state."""

        # Use the configured initial values for reset
        self.entry_count = self.config["initial_counts"].get("entry", 0)
        self.exit_count = self.config["initial_counts"].get("exit", 0)

        self.object_prev_positions = {}
        self.counted_on_entry = set()
        self.counted_on_exit = set()
        self.ct = CentroidTracker(
            max_disappeared=self.config["tracking"]["max_disappeared"]
        )
        self.graph_data = {"timestamps": [], "data_points": []}
        print("Status: Counts and tracking state reset.")

    def add_graph_data(self, current_counts):
        """Adds current count data to the history."""
        # Removed: with self.data_lock:
        self.graph_data["timestamps"].append(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        self.graph_data["data_points"].append(
            {
                "current": current_counts["current_count"],
                "entry": current_counts["entry_count"],
                "exit": current_counts["exit_count"],
            }
        )

    def get_historical_data(self):
        """Returns the stored historical data."""
        # Removed: with self.data_lock:
        return self.graph_data
