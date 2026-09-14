import os
import re
import json
import shutil
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from PyQt5 import QtWidgets, QtCore, QtGui
import sys
from queue import Queue
import threading
import concurrent.futures
import subprocess

GLOBAL_THRESHOLD = 0.5
intervalscan = 4
OUTPUT_DIR = "flagged_frames"
MODEL_PATH = "models/deepdanbooru-v3-20211112-sgd-e28/model-resnet_custom_v3.h5"
TAGS_PATH = "models/deepdanbooru-v3-20211112-sgd-e28/tags.txt"
IMAGE_SIZE = (512, 512)
CONFIG_FILE = "config.json"

NSFW_TAGS = {
 
'nude', 'naked','convenient_censoring','prostitution','between_legs','spread_legs','underwear','breast_squeeze','sex','cleavage','see-through','nipples','midriff','highleg','nudity','rating:questionable','leotard','breasts','lingerie'
}



# === GLOBAL MODEL CACHE ===
_MODEL_CACHE = None
def load_model_once():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _MODEL_CACHE = tf.keras.models.load_model(MODEL_PATH)
    return _MODEL_CACHE


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"threshold": 0.5, "tags": list(NSFW_TAGS),"intervalscan":4}

def save_config(threshold, tags):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold, "tags": tags,"intervalscan":intervalscan}, f, indent=2, ensure_ascii=False)
        


def clear_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  # Delete file or symlink
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Delete directory and its contents
        except Exception as e:
            print(f"Failed to delete {item_path}: {e}")



class ClipGeneratorThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(int, str)

    def __init__(self, flagged_results, video_path, export_dir):
        super().__init__()
        self.flagged_results = flagged_results
        self.video_path = video_path
        self.export_dir = export_dir

    def run(self):
        self.flagged_results.sort(key=lambda x: int(x["timestamp"].replace("s", "")))

        cap = cv2.VideoCapture(self.video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 1
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps
        cap.release()

        clip_dir = os.path.join(self.export_dir, "clips")
        os.makedirs(clip_dir, exist_ok=True)

        clip_length = 4
        clip_gap = 5
        last_clip_time = -clip_gap
        created = 0

        total = len(self.flagged_results)
        tasks = []

        def export_clip(start_time, end_time, out_path):
            print("exporting")
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", f"{start_time:.2f}",
                    "-i", self.video_path,
                    "-t", f"{end_time - start_time:.2f}",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "192k",
                    "-loglevel", "error",
                    out_path
                ],
                check=True
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            for i, entry in enumerate(self.flagged_results):
                t = int(entry["timestamp"].replace("s", ""))
                tag_scores = dict(entry["all_tags"])
                if any(tag in NSFW_TAGS for tag in tag_scores):
                    if t - last_clip_time < clip_gap:
                        continue
                    start_time = max(0, t - clip_length)
                    end_time = min(duration_seconds, t + clip_length)
                    out_path = os.path.join(clip_dir, f"clip_{t}s.mp4")
                    tasks.append(executor.submit(export_clip, start_time, end_time, out_path))
                    created += 1
                    last_clip_time = t
                self.progress.emit(int((i + 1) / total * 100))

        self.finished.emit(created, clip_dir)


class VideoScannerThread(QtCore.QThread):
    progress_changed = QtCore.pyqtSignal(int)
    frame_flagged = QtCore.pyqtSignal(str, object, list, list)
    finished = QtCore.pyqtSignal(int, str, list)

    def __init__(self, video_path, append_mode=False,  parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.append_mode = append_mode
        self._is_running = True
        self.count_flagged = 0
        config = load_config()
        GLOBAL_THRESHOLD  = config["threshold"]
        NSFW_TAGS= "\n".join(config["tags"])
        intervalscan = config["intervalscan"]
        print(f"loaded config with interval {intervalscan}")
        
    def stop(self):
        """Signal the thread to stop."""
        self._is_running = False

    def run(self):
        try:
            model = load_model_once()
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", str(e))
            return

        if not os.path.exists(TAGS_PATH):
            QtWidgets.QMessageBox.critical(None, "Error", "Tags file missing!")
            return

        with open(TAGS_PATH, 'r', encoding='utf-8') as f:
            tags = [line.strip() for line in f]

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        flagged_results = []

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 1
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps

        # Select sampling strategy
        interval = intervalscan
        sample_times = np.arange(0, duration_seconds, interval)

        total = len(sample_times)
        frame_queue = Queue(maxsize=8)

        # === Frame reader thread ===
        def frame_reader():
            for t in sample_times:
                if not self._is_running:
                    break
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ret, frame = cap.read()
                if not ret:
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(frame_rgb, IMAGE_SIZE)
                image_array = resized.astype(np.float32) / 255.0
                
                frame_queue.put((t, image_array, frame.copy()))
                del frame, frame_rgb, resized  # Free memory ASAP
            frame_queue.put(None)

        reader_thread = threading.Thread(target=frame_reader, daemon=True)
        reader_thread.start()

        processed = 0
        batch = []
        meta_times = []
        meta_frames = []
        BATCH_SIZE = 16

        while self._is_running:
            item = frame_queue.get()
            if item is None:
                break
            t, image_array, frame = item
            batch.append(image_array)
            meta_times.append(t)
            meta_frames.append(frame)

            if len(batch) == BATCH_SIZE:
                self._process_batch(model, batch, meta_times,meta_frames, tags, flagged_results)
                processed += len(batch)
                self.progress_changed.emit(int((processed / total) * 100))
                print(f"processed {processed} frames / {total} ({int((processed / total) * 100)}%)")
                batch.clear()
                meta_times.clear()
                meta_frames.clear()

        # Flush last incomplete batch
        if batch and self._is_running:
            self._process_batch(model, batch, meta_times,meta_frames, tags, flagged_results)
            processed += len(batch)
            self.progress_changed.emit(int((processed / total) * 100))

        cap.release()
        reader_thread.join(timeout=1)

        # Save results
        json_path = os.path.join(OUTPUT_DIR, "nsfw_flags.json")
        existing_results = []
        if self.append_mode and os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    existing_results = existing_data.get("results", [])
            except Exception as e:
                print(f"Warning: could not load existing results: {e}")

        deduped_results = self._deduplicate(existing_results + flagged_results)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"video_path": self.video_path, "results": deduped_results},
                      f, indent=2, ensure_ascii=False)

        self.finished.emit(self.count_flagged, json_path, flagged_results)

    def _process_batch(self, model, batch, meta_times,meta_frames, tags, flagged_results):
        preds_batch = model.predict(np.stack(batch), verbose=0)
        for preds, t, frame in zip(preds_batch,meta_times,meta_frames):
            if not self._is_running:
                break
            self._handle_prediction_result(t,frame, preds, tags, flagged_results)

    def _handle_prediction_result(self, t, frame, preds, tags, flagged_results):
        results = list(zip(tags, preds))
        flagged = [(tag, float(score)) for tag, score in results
                   if score >= GLOBAL_THRESHOLD and tag in NSFW_TAGS]
        high_conf_tags = [(tag, float(score)) for tag, score in results
                          if score >= GLOBAL_THRESHOLD]

        if flagged:
            timestamp = int(t)
            tofile = "_".join(tag for tag, _ in flagged)
            filename = f"frame_{tofile}_{timestamp}s.jpg"
            filepath = os.path.join(OUTPUT_DIR, filename)

            # Save JPEG
            # (We don't have the original big frame here, so re-grab it)
            cv2.imwrite(filepath, frame)

            flagged_results.append({
                "timestamp": f"{timestamp}s",
                "frame": filename,
                "nsfw_tags": flagged,
                "all_tags": high_conf_tags
            })

            pixmap = QtGui.QPixmap(filepath).scaled(200, 200, QtCore.Qt.KeepAspectRatio,
                                                    QtCore.Qt.SmoothTransformation)
            self.frame_flagged.emit(filename, pixmap, flagged, high_conf_tags)
            self.count_flagged += 1

    def _deduplicate(self, results):
        seen = set()
        deduped = []
        for item in results:
            key = (item['timestamp'], item['frame'])
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        return deduped



class TimelineFrameWidget(QtWidgets.QWidget):
    clicked = QtCore.pyqtSignal(object)
    def __init__(self, timestamp, pixmap, nsfw_score,filename):
        super().__init__()
        self.timestamp = timestamp
        self.filename = filename

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10,10,10,10)

        label = QtWidgets.QLabel()
        label.setPixmap(
            pixmap.scaled(
                320,
                200,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
        )

        label.setFixedSize(320, 200)
        label.setAlignment(QtCore.Qt.AlignCenter)


        # Style based on NSFW severity
        if nsfw_score >= 0.80:
            border_color = "#ff3333"  # red
        elif nsfw_score >= 0.6:
            border_color = "#ffaa00"  # orange/yellow
        else:
            border_color = "#aaaaaa"  # gray

        label.setStyleSheet(f"border: 2px solid {border_color}; border-radius: 4px;")

        ts_label = QtWidgets.QLabel(f"{timestamp}s")
        ts_label.setAlignment(QtCore.Qt.AlignCenter)
        ts_label.setStyleSheet("color: #ccc; font-size: 10px;")

        layout.addWidget(label)
        layout.addWidget(ts_label)


    def mousePressEvent(self, event):
        self.clicked.emit(self)
     

class VideoScannerApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
          # 1️⃣ Load saved config FIRST
        
        config = load_config()
        global GLOBAL_THRESHOLD, NSFW_TAGS, intervalscan
        GLOBAL_THRESHOLD = config.get("threshold", GLOBAL_THRESHOLD)
        intervalscan = config.get("intervalscan", intervalscan)
        print(intervalscan)
        NSFW_TAGS.clear()
        NSFW_TAGS.update(config.get("tags", list(NSFW_TAGS)))
        print("Loaded the Config file...")
        print(f"tags: {NSFW_TAGS}")
        print(f"Scan interval: {intervalscan}s")
        # 2️⃣ Now build the UI with loaded values
        self.threshold_input = QtWidgets.QDoubleSpinBox()
        self.threshold_input.setRange(0.0, 1.0)
        self.threshold_input.setSingleStep(0.01)
        self.threshold_input.setValue(GLOBAL_THRESHOLD)

        self.tags_input = QtWidgets.QPlainTextEdit()
        self.tags_input.setPlainText("\n".join(NSFW_TAGS))
        self.setWindowTitle("Video NSFW Scanner")
        self.setGeometry(200, 200, 900, 650)
        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
                color: #eeeeee;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            QPushButton {
                background-color: #3a3f4b;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #505661;
            }
           QProgressBar {
    background-color: #1e1e1e;
    border: 1px solid #333;
    border-radius: 8px;
    color: #ddd;
    text-align: center;
    font-weight: bold;
    height: 20px;
}

QProgressBar::chunk {
    background-color: #7a9e3f; /* Olive green */
    border-radius: 8px;
}
            QLabel {
                font-size: 14px;
            }
            QListWidget {
                background-color: #1e1e1e;
                border: none;
            }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        settings_group = QtWidgets.QGroupBox("Settings")
        settings_layout = QtWidgets.QFormLayout(settings_group)

        self.threshold_input = QtWidgets.QDoubleSpinBox()
        self.threshold_input.setRange(0.0, 1.0)
        self.threshold_input.setSingleStep(0.01)
        self.threshold_input.setValue(GLOBAL_THRESHOLD)

        self.tags_input = QtWidgets.QPlainTextEdit()
        self.tags_input.setPlainText("\n".join(NSFW_TAGS))
        self.tags_input.setMaximumHeight(100)
        
        primary_button_style = """
            QPushButton {
                background-color: #3a72f8;
                color: white;
                padding: 6px 14px;
                border-radius: 6px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #295fcc;
            }
        """

        self.btn_apply_settings = QtWidgets.QPushButton("Save Settings")
        self.btn_apply_settings.clicked.connect(self.apply_settings)
        self.btn_apply = QtWidgets.QPushButton("Apply")
        self.btn_apply_settings.setStyleSheet("QPushButton { padding: 5px 12px; }")
        self.btn_apply_settings.setFixedHeight(28)
        settings_layout.addRow("Threshold:", self.threshold_input)
        settings_layout.addRow("Tags (one per line):", self.tags_input)
        settings_layout.addRow(self.btn_apply_settings)

        layout.addWidget(settings_group)


        file_layout = QtWidgets.QHBoxLayout()
        self.btn_pick = QtWidgets.QPushButton("Select Video to Scan")
        self.btn_pick.setStyleSheet(primary_button_style)
        self.btn_pick.setFixedHeight(32)
        self.btn_pick.clicked.connect(self.pick_video)
        self.lbl_file = QtWidgets.QLabel("No file selected")
        file_layout.addWidget(self.btn_pick)
        file_layout.addWidget(self.lbl_file)
        layout.addLayout(file_layout)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)

    
        
        self.timeline_scroll = QtWidgets.QScrollArea()
        self.timeline_scroll.setWidgetResizable(True)
        self.timeline_container = QtWidgets.QWidget()
        self.timeline_layout = QtWidgets.QHBoxLayout(self.timeline_container)
        self.timeline_layout.setSpacing(10)
        self.timeline_layout.setContentsMargins(10, 10, 10, 10)
        self.timeline_scroll.setWidget(self.timeline_container)

        layout.addWidget(QtWidgets.QLabel("Timeline View:"))
        layout.addWidget(self.timeline_scroll)

        self.lbl_info = QtWidgets.QLabel("")
        layout.addWidget(self.lbl_info)

        button_bar = QtWidgets.QHBoxLayout()
        button_bar.setSpacing(8)

        button_style = """
            QPushButton {
                background-color: #2c2f36;
                color: #fff;
                padding: 6px 12px;
                border-radius: 6px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #444a55;
            }
        """

        self.btn_export = QtWidgets.QPushButton("Export")
        self.btn_export.clicked.connect(self.export_selected)
        self.btn_export.setEnabled(False)
        self.btn_export.setStyleSheet(button_style)

        self.btn_heatmap = QtWidgets.QPushButton("Heatmap")
        self.btn_heatmap.clicked.connect(self.show_heatmap)
        self.btn_heatmap.setEnabled(False)
        self.btn_heatmap.setStyleSheet(button_style)

        self.btn_generate_clips = QtWidgets.QPushButton("Clips")
        self.btn_generate_clips.clicked.connect(self.generate_clips)
        self.btn_generate_clips.setEnabled(False)
        self.btn_generate_clips.setStyleSheet(button_style)

        self.btn_rescan_append = QtWidgets.QPushButton("Rescan")
        self.btn_rescan_append.clicked.connect(self.rescan_existing_video)
        self.btn_rescan_append.setEnabled(False)
        self.btn_rescan_append.setStyleSheet(button_style)

        button_bar.addStretch()
        button_bar.addWidget(self.btn_export)
        button_bar.addWidget(self.btn_heatmap)
        button_bar.addWidget(self.btn_generate_clips)
        button_bar.addWidget(self.btn_rescan_append)
        button_bar.addStretch()

        layout.addLayout(button_bar)

        self.scanned_video_path = None  # Store current video path

        self.scanner_thread = None
       

        self.frame_widgets = []
        self.flagged_results = []

    def pick_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.mkv *.webm)")
        if path:
            self.lbl_file.setText(os.path.basename(path))
            self.scanned_video_path = path
            self.flagged_results = []  # clear any previous state
            self.frame_widgets = []
            self.btn_rescan_append.setEnabled(False)  # disable rescan for new video
            self.start_scan(path, append_mode=False)  # start fresh
                
    
    
 
        
    def apply_settings(self):
        global GLOBAL_THRESHOLD, NSFW_TAGS
        GLOBAL_THRESHOLD = self.threshold_input.value()
        tag_lines = [
            tag.strip()
            for tag in self.tags_input.toPlainText().splitlines()
            if tag.strip()
        ]
        NSFW_TAGS.clear()
        NSFW_TAGS.update(tag_lines)

        save_config(GLOBAL_THRESHOLD, list(NSFW_TAGS))
        QtWidgets.QMessageBox.information(
            self,
            "Settings Updated",
            f"Threshold set to {GLOBAL_THRESHOLD}\n"
            f"{len(NSFW_TAGS)} tags saved."
        )

    def remove_timeline_frame(self, widget):
        self.timeline_layout.removeWidget(widget)

        self.flagged_results = [
            x for x in self.flagged_results
            if x["frame"] != widget.filename
        ]

        widget.deleteLater()
    
    def start_scan(self, video_path, append_mode=False):
        self.scanned_video_path = video_path
        self.clear_timeline()
        self.progress.setValue(0)
        self.lbl_info.setText("Starting scan...")
        self.btn_export.setEnabled(False)
        self.frame_widgets = []
        self.flagged_results = []

        if self.scanner_thread and self.scanner_thread.isRunning():
            self.scanner_thread.stop()
            self.scanner_thread.wait()

      

        self.scanner_thread = VideoScannerThread(video_path, append_mode=append_mode)
        self.scanner_thread.progress_changed.connect(self.progress.setValue)
        self.scanner_thread.frame_flagged.connect(self.add_flagged_frame)
        self.scanner_thread.finished.connect(self.scan_finished)
        self.scanner_thread.start()





    def build_analysis_summary_text(self, summary):
        stars = summary["stars"]
        rating = summary["rating"]

        flagged = summary["flagged_count"]
        high_conf = summary["high_confidence_count"]
        very_high = summary["very_high_count"]

        avg_conf = summary["avg_confidence"]
        unique_tags = summary["unique_tags"]

        # Convert 4.2 into:
        # ★ ★ ★ ★ ☆
        full_stars = int(stars)
        half_star = stars - full_stars >= 0.5

        star_text = "★" * full_stars

        if half_star:
            star_text += "½"

        remaining = 5 - full_stars - (1 if half_star else 0)
        star_text += "☆" * remaining

        lines = [
            f"{star_text}  {stars:.1f}/5",
            "",
            rating.upper(),
            "",
            f"Flagged moments: {flagged}",
            f"High-confidence detections: {high_conf}",
            f"Very-high-confidence detections: {very_high}",
            f"Average confidence: {avg_conf:.1%}",
            f"Detected categories: {unique_tags}",
        ]

        # ---------------------------------------------------------
        # Most common tags
        # ---------------------------------------------------------
        tag_counts = summary["tag_counts"]

        if tag_counts:
            lines.extend([
                "",
                "Most detected:"
            ])

            for tag, count in list(tag_counts.items())[:5]:
                lines.append(f"• {tag}: {count}")

        return "\n".join(lines)
    
    
    
    def show_analysis_summary(self, flagged_results):
        summary = self.calculate_analysis_summary(flagged_results)
        text = self.build_analysis_summary_text(summary)

        QtWidgets.QMessageBox.information(
            self,
            "Analysis Summary",
            text
        )
        
    
    def calculate_analysis_summary(self, flagged_results):
        results = flagged_results or []

        if not results:
            return {
                "stars": 0.0,
                "rating": "Clean",
                "flagged_count": 0,
                "high_confidence_count": 0,
                "very_high_count": 0,
                "avg_confidence": 0.0,
                "unique_tags": 0,
                "tag_counts": {},
                "severity_score": 0.0,
            }

        # ---------------------------------------------------------
        # Collect scores and tags
        # ---------------------------------------------------------
        all_scores = []
        tag_counts = {}

        for result in results:
            for tag, score in result.get("all_tags", []):
                score = float(score)
                all_scores.append(score)

                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        flagged_count = len(results)

        high_confidence_count = sum(
            1 for score in all_scores if score >= 0.70
        )

        very_high_count = sum(
            1 for score in all_scores if score >= 0.90
        )

        avg_confidence = (
            sum(all_scores) / len(all_scores)
            if all_scores else 0.0
        )

        unique_tags = len(tag_counts)

        # ---------------------------------------------------------
        # FREQUENCY
        #
        # This is now the dominant factor.
        #
        # 5 flagged moments = still low
        # 10 = mild
        # 20 = moderate
        # 40+ = very frequent
        # ---------------------------------------------------------
        frequency_score = min(flagged_count / 40.0, 1.0)

        # ---------------------------------------------------------
        # CONFIDENCE
        #
        # Confidence should strengthen the rating, but NOT
        # overpower the number of flagged moments.
        # ---------------------------------------------------------
        confidence_score = max(
            0.0,
            (avg_confidence - 0.50) / 0.50
        )

        confidence_score = min(confidence_score, 1.0)

        # ---------------------------------------------------------
        # VERY HIGH CONFIDENCE
        #
        # Small bonus only.
        # ---------------------------------------------------------
        very_high_score = min(
            very_high_count / 20.0,
            1.0
        )

        # ---------------------------------------------------------
        # TAG VARIETY
        #
        # Very small influence.
        # ---------------------------------------------------------
        variety_score = min(
            unique_tags / 5.0,
            1.0
        )

        # ---------------------------------------------------------
        # BASE SCORE
        #
        # Frequency is intentionally dominant.
        # ---------------------------------------------------------
        overall_score = (
            frequency_score * 0.65 +
            confidence_score * 0.20 +
            very_high_score * 0.10 +
            variety_score * 0.05
        )

        # ---------------------------------------------------------
        # HARD LIMITS
        #
        # Prevent a tiny number of detections from producing
        # an exaggerated rating.
        # ---------------------------------------------------------
        if flagged_count <= 2:
            overall_score = min(overall_score, 0.15)

        elif flagged_count <= 5:
            overall_score = min(overall_score, 0.30)

        elif flagged_count <= 10:
            overall_score = min(overall_score, 0.45)

        elif flagged_count <= 20:
            overall_score = min(overall_score, 0.65)

        # ---------------------------------------------------------
        # Clamp
        # ---------------------------------------------------------
        overall_score = max(
            0.0,
            min(overall_score, 1.0)
        )

        stars = round(overall_score * 5, 1)

        # ---------------------------------------------------------
        # Rating
        # ---------------------------------------------------------
        if stars < 0.8:
            rating = "Clean"

        elif stars < 1.6:
            rating = "Very Mild"

        elif stars < 2.4:
            rating = "Mild"

        elif stars < 3.2:
            rating = "Spicy"

        elif stars < 4.2:
            rating = "Very Spicy"

        else:
            rating = "Extremely Spicy"

        # ---------------------------------------------------------
        # Sort tags
        # ---------------------------------------------------------
        tag_counts = dict(
            sorted(
                tag_counts.items(),
                key=lambda item: item[1],
                reverse=True
            )
        )

        return {
            "stars": stars,
            "rating": rating,
            "flagged_count": flagged_count,
            "high_confidence_count": high_confidence_count,
            "very_high_count": very_high_count,
            "avg_confidence": avg_confidence,
            "unique_tags": unique_tags,
            "tag_counts": tag_counts,
            "severity_score": overall_score,
        }

    def add_flagged_frame(self, filename, pixmap, nsfw_tags, all_tags):
        timestamp = int(filename.split("_")[-1].replace("s.jpg", ""))
        max_score = max((score for _, score in nsfw_tags), default=0.0)

        timeline_widget = TimelineFrameWidget(timestamp, pixmap, max_score,filename)

        timeline_widget.clicked.connect(self.remove_timeline_frame)

        self.timeline_layout.addWidget(timeline_widget)
        self.frame_widgets.append(timeline_widget)


    def clear_timeline(self):
        for i in reversed(range(self.timeline_layout.count())):
            self.timeline_layout.itemAt(i).widget().deleteLater()

    def scan_finished(self, count_flagged, json_path, flagged_results):
        self.flagged_results = flagged_results
        self.lbl_info.setText(f"Scan complete. {count_flagged} NSFW frames flagged.\nResults saved to {json_path}")
        
        self.progress.setValue(100)
        self.btn_export.setEnabled(bool(count_flagged))
        self.btn_heatmap.setEnabled(bool(count_flagged))
        self.btn_generate_clips.setEnabled(bool(count_flagged))
        self.btn_rescan_append.setEnabled(True)
        print("========Complete, if slow change interval to a higher value===========")
        self.show_analysis_summary(flagged_results)


    def show_heatmap(self):
        import matplotlib.pyplot as plt
        import numpy as np
        from collections import Counter

        if not self.flagged_results:
            QtWidgets.QMessageBox.information(self, "Heatmap", "No flagged data to visualize.")
            return

        tag_counter = Counter()
        timestamps = []
        all_tag_scores_per_entry = []

        # First pass: collect all tag occurrences and scores
        for entry in self.flagged_results:
            t = int(entry["timestamp"].replace("s", ""))/60
            timestamps.append(t)

            tag_scores = dict(entry["all_tags"])
            all_tag_scores_per_entry.append(tag_scores)

            for tag in tag_scores:
                tag_counter[tag] += 1

        # Sort tags by recurrence
        TOP_N_TAGS = 10
        sorted_tags = [tag for tag, _ in tag_counter.most_common(TOP_N_TAGS)]

        # Build score map initialized with 0.0 for missing tags
        score_map = {tag: [] for tag in sorted_tags}

        for tag_scores in all_tag_scores_per_entry:
            for tag in sorted_tags:
                score_map[tag].append(tag_scores.get(tag, 0.0))

        data = np.array([score_map[tag] for tag in sorted_tags])

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.imshow(data, aspect='auto', cmap='hot', interpolation='nearest')
        ax.set_yticks(np.arange(len(sorted_tags)))
        ax.set_yticklabels(sorted_tags)
        ax.set_xticks(np.arange(len(timestamps)))
        ax.set_xticklabels(timestamps, rotation=45)
        ax.set_xlabel("Time (s)")
        ax.set_title("Tag Confidence Heatmap (Sorted by Frequency)")

        fig.colorbar(cax, ax=ax, label="Confidence")
        plt.tight_layout()
        plt.show()
        
    def rescan_existing_video(self):
        if not self.scanned_video_path or not os.path.exists(self.scanned_video_path):
            QtWidgets.QMessageBox.warning(self, "Missing Video", "Original video path is missing.")
            return

        self.start_scan(self.scanned_video_path, append_mode=True)
    
    def generate_clips(self):
        if not self.flagged_results or not self.scanned_video_path:
            QtWidgets.QMessageBox.warning(self, "Error", "No data or video path.")
            return

        export_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not export_dir:
            return

        self.progress.setValue(0)
        self.lbl_info.setText("Generating clips...")
        self.btn_generate_clips.setEnabled(False)

        self.clip_thread = ClipGeneratorThread(self.flagged_results, self.scanned_video_path, export_dir)
        self.clip_thread.progress.connect(self.progress.setValue)
        self.clip_thread.finished.connect(self.clip_generation_done)
        self.clip_thread.start()

    def clip_generation_done(self, created_count, output_dir):
        self.lbl_info.setText(f"{created_count} clips saved to: {output_dir}")
        self.progress.setValue(100)
        self.btn_generate_clips.setEnabled(True)


    def export_selected(self):
        if not self.flagged_results:
            QtWidgets.QMessageBox.information(self, "Export", "No flagged frames to export.")
            return

        export_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not export_dir:
            return

        exported = 0
        for frame in self.flagged_results:
            src = os.path.join(OUTPUT_DIR, frame['frame'])
            dst = os.path.join(export_dir, frame['frame'])

            if not os.path.exists(src):
                print(f"Warning: File not found, skipping: {src}")
                continue

            try:
                shutil.copy2(src, dst)
                exported += 1
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Export Error", f"Failed to copy {frame['frame']}: {e}")

        # Save JSON metadata
        json_export_path = os.path.join(export_dir, "nsfw_flags_export.json")
        try:
            with open(json_export_path, "w", encoding='utf-8') as f:
                json.dump(self.flagged_results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export Error", f"Failed to save JSON file: {e}")
            return

        QtWidgets.QMessageBox.information(self, "Export Complete",
            f"Exported {exported} flagged frames and metadata to:\n{export_dir}")

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = VideoScannerApp()
    #win.load_previous_results()
    print("clearing last cache..")
    clear_folder("flagged_frames")
    print("ok")
    win.show()
   
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
