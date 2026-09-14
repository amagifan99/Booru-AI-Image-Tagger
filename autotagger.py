import sys
import os
import json
import traceback
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFile

from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import (
    Qt,
    QThread,
    pyqtSignal,
    QMutex,
    QWaitCondition,
    QStringListModel,
)

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
    QFrame,
    QProgressBar,
    QLineEdit,
    QDoubleSpinBox,
    QSplitter,
    QSizePolicy,
    QAbstractItemView,
    QTabWidget,
    QCompleter,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

APP_NAME = "ImageTagger Pro"
APP_VERSION = "2.7"

MODEL_PATH = (
    "models/deepdanbooru-v3-20211112-sgd-e28/"
    "model-resnet_custom_v3.h5"
)

GENERAL_TAGS_PATH = (
    "models/deepdanbooru-v3-20211112-sgd-e28/"
    "tags.txt"
)

IMAGE_SIZE = (512, 512)
DEFAULT_THRESHOLD = 0.40

SUPPORTED_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
)
#will have a different color
NSFW_TAGS = {
    "nude", "naked", "topless", "areola", "nipples",
    "uncensored", "genitals", "pussy", "penis", "cum",
    "sex",
   "raing:explicit",
}


def load_tags(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tag file not found:\n{path}")

    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def prepare_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return image_array[np.newaxis, ...]


def predict_image(model, tags, image_path):
    image_array = prepare_image(image_path)
    predictions = model.predict(image_array, verbose=0)[0]

    results = sorted(
        zip(tags, predictions),
        key=lambda item: item[1],
        reverse=True,
    )

    return [(tag, float(score)) for tag, score in results]


class TaggerThread(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, image_path, model, tags):
        super().__init__()
        self.image_path = image_path
        self.model = model
        self.tags = tags

    def run(self):
        try:
            self.status.emit("Analyzing image...")
            results = predict_image(
                self.model,
                self.tags,
                self.image_path,
            )
            self.finished.emit(results)
        except Exception as exc:
            self.error.emit(
                f"Failed to analyze image:\n{exc}"
            )
            self.finished.emit([])


class BatchTaggerThread(QThread):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int, int)
    status = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, folder_path, model, tags, threshold):
        super().__init__()
        self.folder_path = folder_path
        self.model = model
        self.tags = tags
        self.threshold = threshold

        self._paused = False
        self._stopped = False

        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()

    def pause(self):
        self.mutex.lock()
        self._paused = True
        self.mutex.unlock()

    def resume(self):
        self.mutex.lock()
        self._paused = False
        self.mutex.unlock()
        self.wait_condition.wakeAll()

    def stop(self):
        self.mutex.lock()
        self._stopped = True
        self._paused = False
        self.mutex.unlock()
        self.wait_condition.wakeAll()

    def is_stopped(self):
        self.mutex.lock()
        stopped = self._stopped
        self.mutex.unlock()
        return stopped

    def wait_if_paused(self):
        self.mutex.lock()

        while self._paused and not self._stopped:
            self.wait_condition.wait(self.mutex)

        self.mutex.unlock()

    def find_images(self):
        image_paths = []

        for root, _, files in os.walk(self.folder_path):
            for filename in files:
                if filename.lower().endswith(SUPPORTED_EXTENSIONS):
                    image_paths.append(
                        os.path.join(root, filename)
                    )

        image_paths.sort(key=lambda path: path.lower())
        return image_paths

    def load_existing_results(self, results_path):
        if not os.path.exists(results_path):
            return {}

        try:
            with open(results_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            return data if isinstance(data, dict) else {}

        except Exception as exc:
            self.error.emit(
                f"Could not load existing results:\n{exc}"
            )
            return {}

    def run(self):
        try:
            image_paths = self.find_images()
            total = len(image_paths)

            if total == 0:
                self.error.emit(
                    "No supported images were found."
                )
                self.finished.emit({})
                return

            results_path = os.path.join(
                self.folder_path,
                "tags_results.json",
            )

            results = self.load_existing_results(
                results_path
            )

            processed = 0

            for image_path in image_paths:
                if self.is_stopped():
                    break

                self.wait_if_paused()

                if self.is_stopped():
                    break

                relative_path = os.path.relpath(
                    image_path,
                    self.folder_path,
                )

                if relative_path in results:
                    processed += 1
                    self.progress.emit(processed, total)
                    continue

                self.status.emit(
                    f"Processing: {os.path.basename(image_path)}"
                )

                try:
                    predictions = predict_image(
                        self.model,
                        self.tags,
                        image_path,
                    )

                    filtered = [
                        [tag, score]
                        for tag, score in predictions
                        if score >= self.threshold
                    ]

                    results[relative_path] = filtered

                    with open(
                        results_path,
                        "w",
                        encoding="utf-8",
                    ) as file:
                        json.dump(
                            results,
                            file,
                            indent=2,
                            ensure_ascii=False,
                        )

                except Exception as exc:
                    self.error.emit(
                        f"Failed:\n{relative_path}\n\n{exc}"
                    )

                processed += 1
                self.progress.emit(processed, total)

            self.finished.emit(results)

        except Exception as exc:
            self.error.emit(
                "Batch processing failed:\n"
                f"{exc}\n\n"
                f"{traceback.format_exc()}"
            )
            self.finished.emit({})


class DropLabel(QLabel):
    file_dropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)

    def dragEnterEvent(self, event):
        if not event.mimeData().hasUrls():
            event.ignore()
            return

        for url in event.mimeData().urls():
            path = url.toLocalFile()

            if path.lower().endswith(SUPPORTED_EXTENSIONS):
                event.acceptProposedAction()
                return

        event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()

            if path.lower().endswith(SUPPORTED_EXTENSIONS):
                self.file_dropped.emit(path)
                event.acceptProposedAction()
                return

        event.ignore()


class ImageTagger(QWidget):
    def __init__(self):
        super().__init__()

        self.model = None
        self.tags = []

        self.single_thread = None
        self.batch_thread = None

        self.current_image = None
        self.current_results = []

        self.batch_results = {}
        self.batch_folder = None
        self.current_batch_tags = []

        self.init_window()
        self.load_model()
        self.init_ui()

    def init_window(self):
        self.setWindowTitle(
            f"{APP_NAME}  •  {APP_VERSION}"
        )
        self.setMinimumSize(1250, 800)
        self.resize(1400, 900)

    def load_model(self):
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(
                    f"Model not found:\n{MODEL_PATH}"
                )

            if not os.path.exists(GENERAL_TAGS_PATH):
                raise FileNotFoundError(
                    "General tag file not found:\n"
                    f"{GENERAL_TAGS_PATH}"
                )

            print("Loading DeepDanbooru model...")
            self.model = tf.keras.models.load_model(
                MODEL_PATH
            )

            print("Loading tags...")
            self.tags = load_tags(GENERAL_TAGS_PATH)

            print(
                f"Loaded {len(self.tags):,} tags."
            )

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Startup Error",
                "Could not load the AI model.\n\n"
                f"{exc}",
            )
            raise

    def init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 18)
        root.setSpacing(14)

        header = QHBoxLayout()
        title_layout = QVBoxLayout()

        title = QLabel(APP_NAME)

        title_font = QFont()
        title_font.setPointSize(23)
        title_font.setBold(True)
        title.setFont(title_font)

        subtitle = QLabel(
            "AI-powered image tagging with DeepDanbooru"
        )
        subtitle.setObjectName("subtitle")

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        header.addLayout(title_layout)
        header.addStretch()

        self.batch_button = QPushButton("Batch Folder")
        self.batch_button.clicked.connect(
            self.batch_tag_folder
        )

        self.open_button = QPushButton("Open Image")
        self.open_button.setObjectName("primaryButton")
        self.open_button.clicked.connect(
            self.select_image
        )

        header.addWidget(self.batch_button)
        header.addWidget(self.open_button)

        root.addLayout(header)

        self.tabs = QTabWidget()

        self.single_tab = self.create_single_tab()
        self.batch_tab = self.create_batch_tab()

        self.tabs.addTab(
            self.single_tab,
            "Image Tagger",
        )

        self.tabs.addTab(
            self.batch_tab,
            "Batch Browser",
        )

        root.addWidget(self.tabs, 1)

        status_card = QFrame()
        status_card.setObjectName("statusCard")

        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(
            14, 10, 14, 10
        )

        status_top = QHBoxLayout()

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName(
            "statusLabel"
        )

        status_top.addWidget(self.status_label)
        status_top.addStretch()

        self.pause_button = QPushButton("Pause")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(
            self.pause_batch
        )

        self.resume_button = QPushButton("Resume")
        self.resume_button.setEnabled(False)
        self.resume_button.clicked.connect(
            self.resume_batch
        )

        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(
            self.stop_batch
        )

        status_top.addWidget(self.pause_button)
        status_top.addWidget(self.resume_button)
        status_top.addWidget(self.stop_button)

        status_layout.addLayout(status_top)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        status_layout.addWidget(self.progress_bar)

        root.addWidget(status_card)

    def create_single_tab(self):
        tab = QWidget()

        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 14, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        image_card = QFrame()
        image_card.setObjectName("card")

        image_layout = QVBoxLayout(image_card)
        image_layout.setContentsMargins(
            16, 16, 16, 16
        )

        image_header = QHBoxLayout()

        image_title = QLabel("Preview")
        image_title.setObjectName("sectionTitle")

        image_header.addWidget(image_title)
        image_header.addStretch()

        self.image_info = QLabel(
            "No image selected"
        )
        self.image_info.setObjectName(
            "mutedLabel"
        )

        image_header.addWidget(
            self.image_info
        )

        image_layout.addLayout(image_header)

        self.image_label = DropLabel()
        self.image_label.setObjectName(
            "dropArea"
        )

        self.image_label.setText(
            "Drop an image here\n\n"
            "or\n\n"
            "click  •  Open Image"
        )

        self.image_label.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        self.image_label.file_dropped.connect(
            self.load_image_from_path
        )

        image_layout.addWidget(
            self.image_label,
            1,
        )

        splitter.addWidget(image_card)

        tag_card = QFrame()
        tag_card.setObjectName("card")

        tag_layout = QVBoxLayout(tag_card)
        tag_layout.setContentsMargins(
            16, 16, 16, 16
        )
        tag_layout.setSpacing(10)

        tag_header = QHBoxLayout()

        tag_title = QLabel("Detected Tags")
        tag_title.setObjectName("sectionTitle")

        tag_header.addWidget(tag_title)
        tag_header.addStretch()

        self.tag_count = QLabel("0 tags")
        self.tag_count.setObjectName(
            "mutedLabel"
        )

        tag_header.addWidget(
            self.tag_count
        )

        tag_layout.addLayout(tag_header)

        controls = QHBoxLayout()

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText(
            "Search tags..."
        )

        self.search_box.textChanged.connect(
            self.filter_tags
        )

        controls.addWidget(
            self.search_box,
            1,
        )

        threshold_label = QLabel(
            "Threshold"
        )

        controls.addWidget(
            threshold_label
        )

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(
            0.01, 1.00
        )
        self.threshold_spin.setSingleStep(
            0.05
        )
        self.threshold_spin.setValue(
            DEFAULT_THRESHOLD
        )
        self.threshold_spin.setDecimals(2)

        self.threshold_spin.valueChanged.connect(
            self.refresh_current_results
        )

        controls.addWidget(
            self.threshold_spin
        )

        tag_layout.addLayout(controls)

        self.table = QTableWidget(0, 2)

        self.table.setHorizontalHeaderLabels(
            ["Tag", "Confidence"]
        )

        self.table.horizontalHeader().setSectionResizeMode(
            0,
            QHeaderView.Stretch,
        )

        self.table.horizontalHeader().setSectionResizeMode(
            1,
            QHeaderView.ResizeToContents,
        )

        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)

        self.table.setSelectionBehavior(
            QAbstractItemView.SelectRows
        )

        self.table.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )

        self.table.setSortingEnabled(True)

        tag_layout.addWidget(
            self.table,
            1,
        )

        buttons = QHBoxLayout()

        self.copy_button = QPushButton(
            "Copy Tags"
        )
        self.copy_button.setEnabled(False)
        self.copy_button.clicked.connect(
            self.copy_tags
        )

        self.export_button = QPushButton(
            "Export JSON"
        )
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(
            self.export_single_results
        )

        buttons.addWidget(
            self.copy_button
        )

        buttons.addWidget(
            self.export_button
        )

        buttons.addStretch()

        tag_layout.addLayout(buttons)

        splitter.addWidget(tag_card)
        splitter.setSizes([500, 700])

        layout.addWidget(
            splitter,
            1,
        )

        return tab

    def create_batch_tab(self):
        tab = QWidget()

        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 14, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Left: batch navigation / filtering.
        file_card = QFrame()
        file_card.setObjectName("card")
        file_card.setMinimumWidth(300)

        file_layout = QVBoxLayout(file_card)
        file_layout.setContentsMargins(16, 16, 16, 16)

        file_header = QHBoxLayout()
        file_title = QLabel("Batch Results")
        file_title.setObjectName("sectionTitle")
        file_header.addWidget(file_title)
        file_header.addStretch()

        self.batch_count_label = QLabel("0 images")
        self.batch_count_label.setObjectName("mutedLabel")
        file_header.addWidget(self.batch_count_label)
        file_layout.addLayout(file_header)

        self.batch_search = QLineEdit()
        self.batch_search.setPlaceholderText("Search images...")
        self.batch_search.textChanged.connect(self.filter_batch_files)
        file_layout.addWidget(self.batch_search)

        self.batch_tag_filter = QLineEdit()
        self.batch_tag_filter.setPlaceholderText("Filter images by tag...")
        self.batch_tag_filter.setClearButtonEnabled(True)

        # IMPORTANT: suggestions are built from tags that actually occur
        # in this batch, not from the complete DeepDanbooru vocabulary.
        self.batch_tag_filter_model = QStringListModel([], self)
        self.batch_tag_filter_completer = QCompleter(
            self.batch_tag_filter_model,
            self,
        )
        self.batch_tag_filter_completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.batch_tag_filter_completer.setFilterMode(Qt.MatchContains)
        self.batch_tag_filter_completer.setCompletionMode(QCompleter.PopupCompletion)
        self.batch_tag_filter.setCompleter(self.batch_tag_filter_completer)
        self.batch_tag_filter.textChanged.connect(self.filter_batch_files)
        file_layout.addWidget(self.batch_tag_filter)

        self.batch_file_list = QTableWidget(0, 2)
        self.batch_file_list.setHorizontalHeaderLabels(["Image", "Tags"])
        self.batch_file_list.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.batch_file_list.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self.batch_file_list.verticalHeader().setVisible(False)
        self.batch_file_list.setShowGrid(False)
        self.batch_file_list.setAlternatingRowColors(True)
        self.batch_file_list.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.batch_file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.batch_file_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.batch_file_list.currentCellChanged.connect(self.batch_file_selected)
        file_layout.addWidget(self.batch_file_list, 1)

        batch_buttons = QHBoxLayout()
        self.open_batch_button = QPushButton("Open Results JSON")
        self.open_batch_button.clicked.connect(self.open_batch_results)
        self.export_batch_button = QPushButton("Export")
        self.export_batch_button.clicked.connect(self.export_loaded_batch)
        batch_buttons.addWidget(self.open_batch_button)
        batch_buttons.addWidget(self.export_batch_button)
        file_layout.addLayout(batch_buttons)

        splitter.addWidget(file_card)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Right: image ONLY. No title, tags, confidence, or controls.
        result_card = QFrame()
        result_card.setObjectName("card")
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(0)

        self.batch_preview = QLabel("Select an image from the list")
        self.batch_preview.setObjectName("batchPreview")
        self.batch_preview.setAlignment(Qt.AlignCenter)
        self.batch_preview.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Ignored,
        )
        self.batch_preview.setMinimumSize(0, 0)
        result_layout.addWidget(self.batch_preview, 1)

        splitter.addWidget(result_card)
        splitter.setSizes([380, 900])
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter, 1)
        return tab

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
        )

        if file_path:
            self.load_image_from_path(
                file_path
            )

    def load_image_from_path(self, file_path):
        if not file_path:
            return

        if not os.path.isfile(file_path):
            self.show_error(
                "The selected image does not exist."
            )
            return

        if not file_path.lower().endswith(
            SUPPORTED_EXTENSIONS
        ):
            self.show_error(
                "Unsupported image format."
            )
            return

        self.current_image = file_path

        pixmap = QPixmap(file_path)

        if pixmap.isNull():
            self.show_error(
                "Could not load the selected image."
            )
            return

        self.update_preview(pixmap)

        self.image_info.setText(
            Path(file_path).name
        )

        self.clear_results()
        self.set_single_processing_state(
            True
        )

        self.status_label.setText(
            "Analyzing image..."
        )

        self.single_thread = TaggerThread(
            file_path,
            self.model,
            self.tags,
        )

        self.single_thread.finished.connect(
            self.show_results
        )

        self.single_thread.error.connect(
            self.show_error
        )

        self.single_thread.start()

        self.tabs.setCurrentIndex(0)

    def update_preview(self, pixmap):
        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.image_label.setPixmap(
            scaled
        )

    def update_batch_preview(self, image_path):
        if not image_path:
            self.batch_preview.clear()
            self.batch_preview.setText(
                "Select an image from the list"
            )
            return

        if not os.path.isfile(image_path):
            self.batch_preview.clear()
            self.batch_preview.setText(
                "Image file not found"
            )
            return

        pixmap = QPixmap(image_path)

        if pixmap.isNull():
            self.batch_preview.clear()
            self.batch_preview.setText(
                "Could not load image"
            )
            return

        target_size = self.batch_preview.size()

        if target_size.width() <= 1 or target_size.height() <= 1:
            return

        scaled = pixmap.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.batch_preview.setPixmap(
            scaled
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if self.current_image:
            pixmap = QPixmap(
                self.current_image
            )

            if not pixmap.isNull():
                self.update_preview(
                    pixmap
                )

        current_row = (
            self.batch_file_list.currentRow()
        )

        if current_row >= 0:
            item = self.batch_file_list.item(
                current_row,
                0,
            )

            if item:
                relative_path = item.data(
                    Qt.UserRole
                )

                image_path = self.get_batch_image_path(
                    relative_path
                )

                self.update_batch_preview(
                    image_path
                )

    def show_results(self, results):
        self.current_results = results

        self.refresh_current_results()
        self.set_single_processing_state(
            False
        )

        visible_count = len(
            [
                score
                for _, score in results
                if score >= self.threshold_spin.value()
            ]
        )

        self.status_label.setText(
            "Analysis complete • "
            f"{visible_count} tags above threshold"
        )

    def refresh_current_results(self):
        threshold = (
            self.threshold_spin.value()
        )

        search = (
            self.search_box.text()
            .strip()
            .lower()
        )

        visible_results = []

        for tag, score in self.current_results:
            if score < threshold:
                continue

            if search and search not in tag.lower():
                continue

            visible_results.append(
                (tag, score)
            )

        self.populate_table(
            self.table,
            visible_results,
        )

        total = len(
            [
                score
                for tag, score in self.current_results
                if score >= threshold
            ]
        )

        self.tag_count.setText(
            f"{total} tags"
        )

        self.copy_button.setEnabled(
            bool(visible_results)
        )

        self.export_button.setEnabled(
            bool(self.current_results)
        )

    def filter_tags(self):
        self.refresh_current_results()

    def populate_table(self, table, results):
        table.setSortingEnabled(False)
        table.clearContents()
        table.setRowCount(
            len(results)
        )

        for row, (tag, score) in enumerate(
            results
        ):
            tag_item = QTableWidgetItem(
                tag
            )

            score_item = QTableWidgetItem(
                f"{score * 100:.1f}%"
            )

            score_item.setTextAlignment(
                Qt.AlignRight | Qt.AlignVCenter
            )

            if tag.lower() in NSFW_TAGS:
                tag_item.setForeground(
                    Qt.magenta
                )
                score_item.setForeground(
                    Qt.magenta
                )

            table.setItem(
                row,
                0,
                tag_item,
            )

            table.setItem(
                row,
                1,
                score_item,
            )

        table.setSortingEnabled(True)
        table.sortItems(
            1,
            Qt.DescendingOrder,
        )

    def copy_tags(self):
        if not self.current_results:
            return

        threshold = (
            self.threshold_spin.value()
        )

        search = (
            self.search_box.text()
            .strip()
            .lower()
        )

        tags = []

        for tag, score in self.current_results:
            if score < threshold:
                continue

            if search and search not in tag.lower():
                continue

            tags.append(tag)

        if not tags:
            return

        QApplication.clipboard().setText(
            ", ".join(tags)
        )

        self.status_label.setText(
            f"Copied {len(tags)} tags."
        )

    def export_single_results(self):
        if not self.current_results:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "image_tags.json",
            "JSON Files (*.json)",
        )

        if not path:
            return

        threshold = (
            self.threshold_spin.value()
        )

        filtered = [
            [tag, score]
            for tag, score in self.current_results
            if score >= threshold
        ]

        data = {
            "image": self.current_image,
            "threshold": threshold,
            "tags": filtered,
        }

        try:
            with open(
                path,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    data,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

            self.status_label.setText(
                "Results exported successfully."
            )

        except Exception as exc:
            self.show_error(
                f"Could not export results:\n{exc}"
            )

    def clear_results(self):
        self.current_results = []

        self.table.clearContents()
        self.table.setRowCount(0)

        self.tag_count.setText(
            "0 tags"
        )

        self.copy_button.setEnabled(
            False
        )

        self.export_button.setEnabled(
            False
        )

    def batch_tag_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder",
        )

        if not folder:
            return

        self.batch_folder = folder

        self.set_batch_processing_state(
            True
        )

        self.progress_bar.setValue(0)

        self.status_label.setText(
            "Starting batch processing..."
        )

        self.batch_thread = BatchTaggerThread(
            folder,
            self.model,
            self.tags,
            self.threshold_spin.value(),
        )

        self.batch_thread.progress.connect(
            self.update_progress
        )

        self.batch_thread.status.connect(
            self.update_batch_status
        )

        self.batch_thread.error.connect(
            self.handle_batch_error
        )

        self.batch_thread.finished.connect(
            self.batch_finished
        )

        self.batch_thread.start()

        self.tabs.setCurrentIndex(1)

    def update_progress(self, current, total):
        if total <= 0:
            return

        percentage = int(
            current / total * 100
        )

        self.progress_bar.setValue(
            percentage
        )

        self.status_label.setText(
            f"Processing images • "
            f"{current}/{total}"
        )

    def update_batch_status(self, message):
        self.status_label.setText(
            message
        )

    def pause_batch(self):
        if not self.batch_thread:
            return

        if not self.batch_thread.isRunning():
            return

        self.batch_thread.pause()

        self.pause_button.setEnabled(
            False
        )

        self.resume_button.setEnabled(
            True
        )

        self.status_label.setText(
            "Batch processing paused."
        )

    def resume_batch(self):
        if not self.batch_thread:
            return

        self.batch_thread.resume()

        self.pause_button.setEnabled(
            True
        )

        self.resume_button.setEnabled(
            False
        )

        self.status_label.setText(
            "Resuming batch processing..."
        )

    def stop_batch(self):
        if not self.batch_thread:
            return

        answer = QMessageBox.question(
            self,
            "Stop Batch Processing",
            "Stop the current batch operation?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if answer != QMessageBox.Yes:
            return

        self.batch_thread.stop()

        self.status_label.setText(
            "Stopping batch processing..."
        )

    def batch_finished(self, results):
        self.set_batch_processing_state(
            False
        )

        self.progress_bar.setValue(100)

        self.batch_results = results

        self.populate_batch_browser()

        self.status_label.setText(
            "Batch complete • "
            f"{len(results):,} images"
        )

        self.tabs.setCurrentIndex(1)

    def handle_batch_error(self, message):
        print(
            "Batch warning:",
            message,
        )

    def populate_batch_browser(self):
        self.batch_file_list.clearContents()
        self.batch_file_list.setRowCount(
            0
        )

        self.batch_count_label.setText(
            f"{len(self.batch_results):,} images"
        )

        # Autocomplete ONLY from tags present in the loaded JSON batch.
        # Never use the full DeepDanbooru tag vocabulary here.
        batch_tags = set()
        for image_tags in self.batch_results.values():
            if not isinstance(image_tags, (list, tuple)):
                continue

            for entry in image_tags:
                if isinstance(entry, (list, tuple)) and entry:
                    batch_tags.add(str(entry[0]))
                elif isinstance(entry, str):
                    batch_tags.add(entry)

        self.batch_tag_filter_model.setStringList(
            sorted(batch_tags, key=str.lower)
        )

        for path in sorted(
            self.batch_results.keys(),
            key=lambda value: value.lower(),
        ):
            tags = self.batch_results[path]

            row = self.batch_file_list.rowCount()

            self.batch_file_list.insertRow(
                row
            )

            image_item = QTableWidgetItem(
                os.path.basename(path)
            )

            image_item.setToolTip(path)
            image_item.setData(
                Qt.UserRole,
                path,
            )

            tag_item = QTableWidgetItem(
                str(len(tags))
            )

            tag_item.setTextAlignment(
                Qt.AlignRight | Qt.AlignVCenter
            )

            self.batch_file_list.setItem(
                row,
                0,
                image_item,
            )

            self.batch_file_list.setItem(
                row,
                1,
                tag_item,
            )

        if self.batch_file_list.rowCount():
            self.batch_file_list.setCurrentCell(
                0,
                0,
            )
        else:
            self.clear_batch_selection()

        self.filter_batch_files()

    def filter_batch_files(self, text=""):
        image_search = (
            self.batch_search.text()
            .strip()
            .lower()
        )

        tag_search = (
            self.batch_tag_filter.text()
            .strip()
            .lower()
        )

        requested_tags = [
            tag.strip()
            for tag in tag_search.split(",")
            if tag.strip()
        ]

        visible_count = 0
        total = self.batch_file_list.rowCount()

        for row in range(total):
            item = self.batch_file_list.item(
                row,
                0,
            )

            if not item:
                continue

            path = item.data(Qt.UserRole)

            filename_match = (
                not image_search
                or image_search in path.lower()
            )

            image_tags = self.batch_results.get(
                path,
                []
            )

            available_tags = {
                str(tag).lower()
                for tag, _ in image_tags
            }

            tag_match = all(
                any(
                    requested in available
                    for available in available_tags
                )
                for requested in requested_tags
            )

            visible = filename_match and tag_match

            self.batch_file_list.setRowHidden(
                row,
                not visible,
            )

            if visible:
                visible_count += 1

        if image_search or requested_tags:
            self.batch_count_label.setText(
                f"{visible_count:,} / {total:,} images"
            )
        else:
            self.batch_count_label.setText(
                f"{total:,} images"
            )

    def get_batch_image_path(self, relative_path):
        if os.path.isabs(relative_path):
            return relative_path

        if not self.batch_folder:
            return relative_path

        return os.path.join(
            self.batch_folder,
            relative_path,
        )

    def batch_file_selected(
        self,
        current_row,
        current_column,
        previous_row,
        previous_column,
    ):
        if current_row < 0:
            self.clear_batch_selection()
            return

        item = self.batch_file_list.item(
            current_row,
            0,
        )

        if not item:
            self.clear_batch_selection()
            return

        relative_path = item.data(
            Qt.UserRole
        )

        image_path = self.get_batch_image_path(
            relative_path
        )

        # Batch Browser is viewer-only on the right.
        # Do not create/update any tags or confidence widgets here.
        self.update_batch_preview(image_path)

    def clear_batch_selection(self):
        self.batch_preview.clear()
        self.batch_preview.setText(
            "Select an image from the list"
        )

    def open_batch_results(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Batch Results",
            "",
            "JSON Files (*.json)",
        )

        if not path:
            return

        try:
            with open(
                path,
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

            if not isinstance(data, dict):
                raise ValueError(
                    "Invalid batch results format."
                )

            self.batch_results = data
            self.batch_folder = os.path.dirname(
                path
            )

            self.populate_batch_browser()
            self.tabs.setCurrentIndex(1)

            self.status_label.setText(
                f"Loaded {len(data):,} batch results."
            )

        except Exception as exc:
            self.show_error(
                f"Could not open results:\n{exc}"
            )

    def export_loaded_batch(self):
        if not self.batch_results:
            QMessageBox.information(
                self,
                "Batch Browser",
                "There are no batch results loaded.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Batch Results",
            "tags_results.json",
            "JSON Files (*.json)",
        )

        if not path:
            return

        try:
            with open(
                path,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    self.batch_results,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

            self.status_label.setText(
                "Batch results exported."
            )

        except Exception as exc:
            self.show_error(
                f"Could not export results:\n{exc}"
            )

    def set_single_processing_state(
        self,
        processing,
    ):
        self.open_button.setEnabled(
            not processing
        )

        self.batch_button.setEnabled(
            not processing
        )

        self.threshold_spin.setEnabled(
            not processing
        )

    def set_batch_processing_state(
        self,
        processing,
    ):
        self.open_button.setEnabled(
            not processing
        )

        self.batch_button.setEnabled(
            not processing
        )

        self.threshold_spin.setEnabled(
            not processing
        )

        self.pause_button.setEnabled(
            processing
        )

        self.resume_button.setEnabled(
            False
        )

        self.stop_button.setEnabled(
            processing
        )

    def show_error(self, message):
        QMessageBox.warning(
            self,
            APP_NAME,
            message,
        )

    def closeEvent(self, event):
        if (
            self.batch_thread
            and self.batch_thread.isRunning()
        ):
            answer = QMessageBox.question(
                self,
                "Batch Processing",
                "A batch operation is still running.\n\n"
                "Stop it and exit?",
                QMessageBox.Yes | QMessageBox.No,
            )

            if answer != QMessageBox.Yes:
                event.ignore()
                return

            self.batch_thread.stop()
            self.batch_thread.wait(3000)

        if (
            self.single_thread
            and self.single_thread.isRunning()
        ):
            self.single_thread.quit()
            self.single_thread.wait(2000)

        event.accept()


APP_STYLE = """
QWidget {
    background-color: #11111b;
    color: #cdd6f4;
    font-family: "Segoe UI";
    font-size: 14px;
}

QFrame#card {
    background-color: #181825;
    border: 1px solid #313244;
    border-radius: 14px;
}

QFrame#statusCard {
    background-color: #181825;
    border: 1px solid #313244;
    border-radius: 10px;
}

QFrame#previewFrame {
    background-color: #11111b;
    border: 1px solid #313244;
    border-radius: 10px;
}

QLabel {
    color: #cdd6f4;
}

QLabel#subtitle {
    color: #7f849c;
    font-size: 13px;
}

QLabel#sectionTitle {
    color: #f5f5f5;
    font-size: 16px;
    font-weight: 700;
}

QLabel#mutedLabel {
    color: #7f849c;
    font-size: 12px;
}

QLabel#statusLabel {
    color: #a6adc8;
    font-weight: 600;
}

QLabel#dropArea {
    background-color: #11111b;
    border: 2px dashed #45475a;
    border-radius: 12px;
    color: #7f849c;
    font-size: 14px;
}

QLabel#dropArea:hover {
    border-color: #89b4fa;
    color: #bac2de;
}

QLabel#batchPreview {
    background-color: #11111b;
    color: #7f849c;
    border-radius: 8px;
}

QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 9px;
    padding: 9px 15px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #45475a;
    border-color: #585b70;
}

QPushButton:pressed {
    background-color: #585b70;
}

QPushButton:disabled {
    background-color: #1e1e2e;
    color: #585b70;
    border-color: #313244;
}

QPushButton#primaryButton {
    background-color: #89b4fa;
    color: #11111b;
    border: none;
}

QPushButton#primaryButton:hover {
    background-color: #b4befe;
}

QLineEdit {
    background-color: #11111b;
    border: 1px solid #313244;
    border-radius: 8px;
    padding: 8px 10px;
    color: #cdd6f4;
}

QLineEdit:focus {
    border-color: #89b4fa;
}

QDoubleSpinBox {
    background-color: #11111b;
    border: 1px solid #313244;
    border-radius: 8px;
    padding: 7px;
    color: #cdd6f4;
}

QDoubleSpinBox:focus {
    border-color: #89b4fa;
}

QTabWidget::pane {
    border: none;
}

QTabBar::tab {
    background-color: transparent;
    color: #7f849c;
    padding: 9px 18px;
    margin-right: 4px;
    border-bottom: 2px solid transparent;
}

QTabBar::tab:hover {
    color: #cdd6f4;
}

QTabBar::tab:selected {
    color: #89b4fa;
    border-bottom: 2px solid #89b4fa;
}

QTableWidget {
    background-color: transparent;
    border: none;
    outline: none;
    alternate-background-color: #1e1e2e;
    gridline-color: transparent;
}

QTableWidget::item {
    padding: 9px;
    border: none;
}

QTableWidget::item:selected {
    background-color: #313244;
    color: #f5f5f5;
}

QHeaderView::section {
    background-color: transparent;
    color: #a6adc8;
    border: none;
    border-bottom: 1px solid #313244;
    padding: 8px;
    font-weight: 700;
}

QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #313244;
    border-radius: 5px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background: #45475a;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background: transparent;
    height: 10px;
}

QScrollBar::handle:horizontal {
    background: #313244;
    border-radius: 5px;
}

QProgressBar {
    background-color: #11111b;
    border: none;
    border-radius: 5px;
    height: 8px;
    text-align: center;
    color: #cdd6f4;
}

QProgressBar::chunk {
    background-color: #89b4fa;
    border-radius: 5px;
}

QSplitter::handle {
    background-color: #11111b;
    width: 6px;
}
"""


def main():
    app = QApplication(sys.argv)

    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setStyleSheet(APP_STYLE)

    try:
        window = ImageTagger()
    except Exception:
        sys.exit(1)

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
