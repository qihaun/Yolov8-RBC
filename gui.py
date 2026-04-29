import sys
import os
import traceback
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QCheckBox, QLabel, QFileDialog, QSlider, QDoubleSpinBox,
                             QGroupBox, QMessageBox, QScrollArea, QFrame, QSizePolicy,
                             QProgressBar)
from PyQt5.QtGui import QImage, QPixmap, QFont
from ultralytics import YOLO


# ==================== 全局样式表 ====================
STYLE_QSS = """
/* 主窗口 */
QMainWindow {
    background-color: #f0f4f8;
}

/* 右侧图片滚动区域 */
QScrollArea {
    border: none;
    background: transparent;
}

/* 分组框 */
QGroupBox {
    font-size: 12px;
    font-weight: bold;
    color: #1a365d;
    border: 1px solid #c8d6e5;
    border-radius: 6px;
    margin-top: 10px;
    padding: 10px 6px 6px 6px;
    background-color: #ffffff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 5px;
    color: #2c5282;
}

/* 普通按钮 */
QPushButton {
    border: none;
    border-radius: 8px;
    padding: 7px 16px;
    font-size: 12px;
    font-weight: bold;
    color: #ffffff;
    background-color: #3182ce;
}
QPushButton:hover {
    background-color: #2b6cb0;
}
QPushButton:pressed {
    background-color: #2c5282;
}
QPushButton:disabled {
    background-color: #a0aec0;
    color: #e2e8f0;
}

/* 主要操作按钮 */
QPushButton#btn_detect {
    background-color: #e53e3e;
    font-size: 15px;
    font-weight: bold;
    padding: 10px;
}
QPushButton#btn_detect:hover {
    background-color: #c53030;
}
QPushButton#btn_detect:disabled {
    background-color: #a0aec0;
}

/* 模型加载按钮 */
QPushButton#btn_load_model {
    background-color: #38a169;
}
QPushButton#btn_load_model:hover {
    background-color: #2f855a;
}

/* 清空按钮 */
QPushButton#btn_clear {
    background-color: #718096;
}
QPushButton#btn_clear:hover {
    background-color: #4a5568;
}

/* 导航按钮 */
QPushButton#btn_prev, QPushButton#btn_next {
    background-color: #edf2f7;
    color: #2d3748;
    font-weight: bold;
}
QPushButton#btn_prev:hover, QPushButton#btn_next:hover {
    background-color: #e2e8f0;
}

/* 复选框 */
QCheckBox {
    font-size: 12px;
    color: #2d3748;
    spacing: 6px;
    padding: 2px 0;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 3px;
    border: 2px solid #a0aec0;
    background-color: #ffffff;
}
QCheckBox::indicator:checked {
    background-color: #3182ce;
    border-color: #3182ce;
}

/* 滑块 */
QSlider::groove:horizontal {
    border-radius: 3px;
    height: 5px;
    background-color: #e2e8f0;
}
QSlider::handle:horizontal {
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
    background-color: #3182ce;
    border: 2px solid #ffffff;
}
QSlider::sub-page:horizontal {
    background-color: #3182ce;
    border-radius: 4px;
}

/* 数值框 */
QDoubleSpinBox {
    border: 1px solid #c8d6e5;
    border-radius: 4px;
    padding: 3px 6px;
    font-size: 12px;
    background-color: #ffffff;
    min-width: 55px;
}
QDoubleSpinBox:focus {
    border-color: #3182ce;
}

/* 进度条 */
QProgressBar {
    border: none;
    border-radius: 6px;
    height: 8px;
    background-color: #e2e8f0;
    text-align: center;
    font-size: 11px;
    color: #4a5568;
}
QProgressBar::chunk {
    border-radius: 6px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #3182ce, stop:1 #63b3ed);
}

/* 标签 */
QLabel {
    color: #2d3748;
}

/* 状态标签 */
QLabel#status_label {
    font-size: 12px;
    color: #4a5568;
    padding: 4px 10px;
    background-color: #edf2f7;
    border-radius: 4px;
}

/* 统计标签 */
QLabel#stat_label {
    font-size: 12px;
    padding: 6px;
    background-color: #f7fafc;
    border-radius: 6px;
}

/* 页码标签 */
QLabel#page_label {
    font-size: 13px;
    font-weight: bold;
    color: #2c5282;
}

/* 模型标签 */
QLabel#model_label {
    font-size: 11px;
    color: #4a5568;
    padding: 3px 6px;
    background-color: #ebf8ff;
    border-radius: 4px;
}
"""


class DetectThread(QThread):
    finished_signal = pyqtSignal(list)
    progress_signal = pyqtSignal(int, int)
    error_signal = pyqtSignal(str)

    def __init__(self, model, img_paths, conf_thres, iou_thres, show_rbc, show_wbc, show_plate):
        super().__init__()
        self.model = model
        self.img_paths = img_paths
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.show_rbc = show_rbc
        self.show_wbc = show_wbc
        self.show_plate = show_plate

        self.class_map = {
            0: "RBC",
            1: "WBC",
            2: "Platelet"
        }

    def run(self):
        try:
            results = []
            total = len(self.img_paths)
            for i, path in enumerate(self.img_paths):
                pred = self.model.predict(
                    source=path,
                    conf=self.conf_thres,
                    iou=self.iou_thres,
                    save=False,
                    verbose=False
                )[0]

                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                filtered_boxes = []
                for box in pred.boxes:
                    cls_id = int(box.cls)
                    cls_name = self.class_map.get(cls_id, "unknown")

                    if (cls_name == "RBC" and self.show_rbc) or \
                            (cls_name == "WBC" and self.show_wbc) or \
                            (cls_name == "Platelet" and self.show_plate):
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf)
                        filtered_boxes.append([x1, y1, x2, y2, cls_name, conf])

                for x1, y1, x2, y2, cls_name, conf in filtered_boxes:
                    if cls_name == "RBC":
                        color = (220, 50, 50)
                    elif cls_name == "WBC":
                        color = (50, 180, 80)
                    else:
                        color = (60, 100, 220)

                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, f"{cls_name} {conf:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                results.append((path, img, filtered_boxes))
                self.progress_signal.emit(i + 1, total)

            self.finished_signal.emit(results)
        except Exception as e:
            self.error_signal.emit(str(e))


class RBCDetectWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("血细胞检测系统 — YOLOv8")
        self.setGeometry(100, 100, 1700, 950)
        self.setMinimumSize(1300, 750)

        self.model_path = None
        self.model = None

        self.current_img_paths = []
        self.current_results = []
        self.current_idx = 0
        self.show_rbc = True
        self.show_wbc = True
        self.show_plate = True
        self.detect_running = False

        self.init_ui()
        self.apply_theme()

    def apply_theme(self):
        self.setStyleSheet(STYLE_QSS)

    def load_model(self, path):
        print(f"[DEBUG] 正在加载模型: {path}")
        try:
            self.model = YOLO(path)
            print(f"[DEBUG] 模型加载成功")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "加载失败", f"模型加载失败：\n{str(e)}\n\n路径：{path}")
            return

        self.model_path = os.path.abspath(path)
        self.model_label.setText(os.path.basename(path))
        self.model_label.setStyleSheet(
            "color: #22543d; background-color: #c6f6d5; border-radius: 4px; padding: 3px 6px; font-size: 11px;"
        )
        self.status_indicator.setText("模型已就绪")
        self.status_indicator.setStyleSheet(
            "color: #38a169; font-size: 11px; font-weight: bold; padding: 2px 0;"
        )

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # ========== 左侧控制面板（可上下滚动）==========
        control_widget = QWidget()
        control_widget.setObjectName("control_panel")
        control_widget.setMinimumWidth(400)
        control_widget.setMaximumWidth(500)
        control_layout = QVBoxLayout(control_widget)
        control_layout.setSpacing(6)
        control_layout.setContentsMargins(4, 0, 4, 0)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        left_scroll.setWidget(control_widget)

        # --- 0. 标题栏 ---
        header_frame = QFrame()
        header_frame.setStyleSheet(
            "QFrame { background-color: #ffffff; border-radius: 8px; border: 1px solid #e2e8f0; padding: 8px; }"
        )
        header_layout = QVBoxLayout(header_frame)
        header_layout.setSpacing(3)
        header_layout.setContentsMargins(10, 6, 10, 6)
        title = QLabel("血细胞检测系统")
        title.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        title.setStyleSheet("color: #1a365d; border: none;")
        self.status_indicator = QLabel("未加载模型")
        self.status_indicator.setStyleSheet("color: #a0aec0; font-size: 11px; border: none;")
        header_layout.addWidget(title)
        header_layout.addWidget(self.status_indicator)
        control_layout.addWidget(header_frame)

        # --- 1. 模型管理 ---
        model_group = QGroupBox("模型管理")
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(5)
        self.model_label = QLabel("未加载模型")
        self.model_label.setObjectName("model_label")
        self.model_label.setWordWrap(True)
        self.btn_load_model = QPushButton("加载模型文件")
        self.btn_load_model.setObjectName("btn_load_model")
        self.btn_load_model.clicked.connect(self.select_model)
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.btn_load_model)
        control_layout.addWidget(model_group)

        default_model = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "runs", "detect", "train7", "weights", "best.pt"
        )
        if os.path.exists(default_model):
            self.load_model(default_model)
        else:
            self.model_label.setText("未找到默认模型，请手动加载")
            self.model_label.setStyleSheet(
                "color: #9b2c2c; background-color: #fed7d7; border-radius: 4px; padding: 3px 6px; font-size: 11px;"
            )
            self.status_indicator.setText("未加载模型")

        # --- 2. 类别显示控制 ---
        class_group = QGroupBox("类别显示控制")
        class_layout = QVBoxLayout(class_group)
        class_layout.setSpacing(0)
        self.cb_rbc = QCheckBox("红细胞 (RBC)")
        self.cb_wbc = QCheckBox("白细胞 (WBC)")
        self.cb_plate = QCheckBox("血小板 (Platelet)")
        self.cb_rbc.setChecked(True)
        self.cb_wbc.setChecked(True)
        self.cb_plate.setChecked(True)
        self.cb_rbc.setStyleSheet(
            "QCheckBox { color: #c53030; font-weight: bold; font-size: 12px; padding: 2px 0; }"
            "QCheckBox::indicator:checked { background-color: #c53030; border-color: #c53030; }"
        )
        self.cb_wbc.setStyleSheet(
            "QCheckBox { color: #276749; font-weight: bold; font-size: 12px; padding: 2px 0; }"
            "QCheckBox::indicator:checked { background-color: #38a169; border-color: #38a169; }"
        )
        self.cb_plate.setStyleSheet(
            "QCheckBox { color: #2b6cb0; font-weight: bold; font-size: 12px; padding: 2px 0; }"
            "QCheckBox::indicator:checked { background-color: #3182ce; border-color: #3182ce; }"
        )
        class_layout.addWidget(self.cb_rbc)
        class_layout.addWidget(self.cb_wbc)
        class_layout.addWidget(self.cb_plate)
        control_layout.addWidget(class_group)

        # --- 3. 阈值设置 ---
        thresh_group = QGroupBox("阈值设置")
        thresh_layout = QVBoxLayout(thresh_group)
        thresh_layout.setSpacing(5)
        # 置信度
        conf_row = QHBoxLayout()
        conf_row.setSpacing(5)
        conf_label = QLabel("Conf")
        conf_label.setFixedWidth(30)
        conf_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #4a5568; border: none;")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(25)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setSingleStep(0.01)
        conf_row.addWidget(conf_label)
        conf_row.addWidget(self.conf_slider)
        conf_row.addWidget(self.conf_spin)
        thresh_layout.addLayout(conf_row)
        # IoU
        iou_row = QHBoxLayout()
        iou_row.setSpacing(5)
        iou_label = QLabel("IoU")
        iou_label.setFixedWidth(30)
        iou_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #4a5568; border: none;")
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(1, 100)
        self.iou_slider.setValue(45)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setValue(0.45)
        self.iou_spin.setSingleStep(0.01)
        iou_row.addWidget(iou_label)
        iou_row.addWidget(self.iou_slider)
        iou_row.addWidget(self.iou_spin)
        thresh_layout.addLayout(iou_row)
        control_layout.addWidget(thresh_group)

        # --- 4. 图片导入 ---
        img_group = QGroupBox("图片导入")
        img_layout = QVBoxLayout(img_group)
        img_layout.setSpacing(5)
        self.btn_single = QPushButton("导入单张图片")
        self.btn_folder = QPushButton("导入文件夹")
        img_layout.addWidget(self.btn_single)
        img_layout.addWidget(self.btn_folder)
        control_layout.addWidget(img_group)

        # --- 5. 操作按钮 ---
        action_group = QGroupBox("操作")
        action_layout = QVBoxLayout(action_group)
        action_layout.setSpacing(5)
        self.btn_detect = QPushButton("开 始 检 测")
        self.btn_detect.setObjectName("btn_detect")
        self.btn_save = QPushButton("保存检测结果")
        self.btn_clear = QPushButton("清空显示")
        self.btn_clear.setObjectName("btn_clear")
        action_layout.addWidget(self.btn_detect)
        action_layout.addWidget(self.btn_save)
        action_layout.addWidget(self.btn_clear)
        control_layout.addWidget(action_group)

        # --- 6. 进度条 ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_label = QLabel("")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("color: #718096; font-size: 11px; border: none;")
        self.progress_label.setVisible(False)
        control_layout.addWidget(self.progress_bar)
        control_layout.addWidget(self.progress_label)

        # --- 7. 图片导航 ---
        nav_group = QGroupBox("图片导航")
        nav_layout = QVBoxLayout(nav_group)
        nav_layout.setSpacing(5)
        nav_btn_row = QHBoxLayout()
        nav_btn_row.setSpacing(6)
        self.btn_prev = QPushButton("上一张")
        self.btn_prev.setObjectName("btn_prev")
        self.btn_next = QPushButton("下一张")
        self.btn_next.setObjectName("btn_next")
        self.page_label = QLabel("0 / 0")
        self.page_label.setObjectName("page_label")
        self.page_label.setAlignment(Qt.AlignCenter)
        nav_layout.addLayout(nav_btn_row)
        nav_layout.addWidget(self.page_label)
        control_layout.addWidget(nav_group)

        # --- 8. 统计信息 ---
        stat_group = QGroupBox("检测统计")
        stat_layout = QVBoxLayout(stat_group)
        self.stat_label = QLabel("暂无数据")
        self.stat_label.setObjectName("stat_label")
        self.stat_label.setWordWrap(True)
        stat_layout.addWidget(self.stat_label)
        control_layout.addWidget(stat_group)

        control_layout.addStretch()

        # ========== 右侧显示区域 ==========
        display_panel = QFrame()
        display_panel.setStyleSheet(
            "QFrame { background-color: #ffffff; border-radius: 10px; border: 2px solid #d0d7de; }"
        )
        display_layout = QVBoxLayout(display_panel)
        display_layout.setContentsMargins(8, 8, 8, 8)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setAlignment(Qt.AlignCenter)
        self.scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.img_label = QLabel("\n\n\n请导入图片进行检测\n\n支持 JPG / PNG / BMP / TIFF 格式\n\n\n")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.img_label.setMinimumSize(500, 350)
        self.img_label.setStyleSheet(
            "color: #a0aec0; font-size: 14px; border: 2px dashed #e2e8f0; "
            "border-radius: 12px; background-color: #fafbfc;"
        )
        self.scroll.setWidget(self.img_label)
        display_layout.addWidget(self.scroll)

        # 底部状态栏
        status_bar = QFrame()
        status_bar.setStyleSheet(
            "QFrame { background-color: #edf2f7; border-radius: 6px; }"
        )
        status_bar_layout = QHBoxLayout(status_bar)
        status_bar_layout.setContentsMargins(12, 6, 12, 6)
        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("status_label")
        self.img_count_label = QLabel("")
        self.img_count_label.setStyleSheet("color: #a0aec0; font-size: 11px; border: none;")
        status_bar_layout.addWidget(self.status_label)
        status_bar_layout.addStretch()
        status_bar_layout.addWidget(self.img_count_label)
        display_layout.addWidget(status_bar)

        main_layout.addWidget(left_scroll, stretch=1)
        main_layout.addWidget(display_panel, stretch=4)

        # ========== 信号绑定 ==========
        self.conf_slider.valueChanged.connect(self._on_conf_slider_changed)
        self.conf_spin.valueChanged.connect(self._on_conf_spin_changed)
        self.iou_slider.valueChanged.connect(self._on_iou_slider_changed)
        self.iou_spin.valueChanged.connect(self._on_iou_spin_changed)

        self.btn_single.clicked.connect(self.load_single_img)
        self.btn_folder.clicked.connect(self.load_folder_imgs)
        self.btn_detect.clicked.connect(self.start_detect)
        self.btn_save.clicked.connect(self.save_results)
        self.btn_clear.clicked.connect(self.clear_display)
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)

    # ========== 滑块联动 ==========
    def _on_conf_slider_changed(self, v):
        val = v / 100.0
        self.conf_spin.blockSignals(True)
        self.conf_spin.setValue(val)
        self.conf_spin.blockSignals(False)

    def _on_conf_spin_changed(self, v):
        self.conf_slider.blockSignals(True)
        self.conf_slider.setValue(int(v * 100))
        self.conf_slider.blockSignals(False)

    def _on_iou_slider_changed(self, v):
        val = v / 100.0
        self.iou_spin.blockSignals(True)
        self.iou_spin.setValue(val)
        self.iou_spin.blockSignals(False)

    def _on_iou_spin_changed(self, v):
        self.iou_slider.blockSignals(True)
        self.iou_slider.setValue(int(v * 100))
        self.iou_slider.blockSignals(False)

    # ========== 功能函数 ==========
    def select_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "",
            "PyTorch 模型 (*.pt *.pth);;所有文件 (*.*)"
        )
        if path:
            self.load_model(path)

    def load_single_img(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"
        )
        if path:
            self.current_img_paths = [path]
            self.current_results = []
            self.current_idx = 0
            self.show_image_at_idx(0)
            self.status_label.setText(f"已导入：{os.path.basename(path)}")
            self.img_count_label.setText("1 张图片")

    def load_folder_imgs(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder:
            return
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        paths = []
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(exts):
                paths.append(os.path.join(folder, f))
        if not paths:
            QMessageBox.warning(self, "提示", "所选文件夹中没有图片文件！")
            return
        self.current_img_paths = paths
        self.current_results = []
        self.current_idx = 0
        self.show_image_at_idx(0)
        self.status_label.setText(f"已导入文件夹：{os.path.basename(folder)}")
        self.img_count_label.setText(f"{len(paths)} 张图片")

    def show_image_at_idx(self, idx):
        if 0 <= idx < len(self.current_img_paths):
            img = cv2.imread(self.current_img_paths[idx])
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.display_img(img)
                self.update_page_label()

    def update_page_label(self):
        total = len(self.current_img_paths)
        if total > 0:
            cur = self.current_idx + 1 if self.current_idx < total else 0
            self.page_label.setText(f"{cur} / {total}")
        else:
            self.page_label.setText("0 / 0")

    def prev_image(self):
        total = len(self.current_results) if self.current_results else len(self.current_img_paths)
        if total == 0:
            return
        self.current_idx = (self.current_idx - 1) % total
        if self.current_results:
            _, img, boxes = self.current_results[self.current_idx]
            self.display_img(img)
            self.update_stat_label(boxes)
        else:
            self.show_image_at_idx(self.current_idx)
        self.update_page_label()

    def next_image(self):
        total = len(self.current_results) if self.current_results else len(self.current_img_paths)
        if total == 0:
            return
        self.current_idx = (self.current_idx + 1) % total
        if self.current_results:
            _, img, boxes = self.current_results[self.current_idx]
            self.display_img(img)
            self.update_stat_label(boxes)
        else:
            self.show_image_at_idx(self.current_idx)
        self.update_page_label()

    def display_img(self, img):
        if img is None:
            return
        h, w, ch = img.shape
        bytes_per_line = ch * w
        img_copy = img.copy()
        qt_img = QImage(img_copy.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        scaled = pixmap.scaled(
            self.img_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.img_label.setPixmap(scaled)
        self.img_label.setStyleSheet("border: none; background-color: transparent;")
        self._current_qt_img = qt_img
        self._current_img_copy = img_copy

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_current_img_copy') and self._current_img_copy is not None:
            self.display_img(self._current_img_copy)

    def update_stat_label(self, boxes):
        rbc_count = sum(1 for b in boxes if b[4] == "RBC")
        wbc_count = sum(1 for b in boxes if b[4] == "WBC")
        plt_count = sum(1 for b in boxes if b[4] == "Platelet")
        self.stat_label.setText(
            f"<div style='line-height:1.6;'>"
            f"<p><span style='color:#c53030;font-weight:bold;'>RBC</span>  "
            f"<span style='font-size:16px;font-weight:bold;color:#c53030;'>{rbc_count}</span>  个</p>"
            f"<p><span style='color:#38a169;font-weight:bold;'>WBC</span>  "
            f"<span style='font-size:16px;font-weight:bold;color:#38a169;'>{wbc_count}</span>  个</p>"
            f"<p><span style='color:#3182ce;font-weight:bold;'>Platelet</span>  "
            f"<span style='font-size:16px;font-weight:bold;color:#3182ce;'>{plt_count}</span>  个</p>"
            f"<hr style='border:1px solid #e2e8f0;'>"
            f"<p style='font-weight:bold;'>总计：{len(boxes)} 个</p>"
            f"</div>"
        )

    def start_detect(self):
        if not self.model:
            QMessageBox.warning(self, "提示", "请先加载模型！")
            return
        if not self.current_img_paths:
            QMessageBox.warning(self, "提示", "请先导入图片！")
            return
        if self.detect_running:
            QMessageBox.warning(self, "提示", "检测正在进行中，请稍候...")
            return

        conf = self.conf_spin.value()
        iou = self.iou_spin.value()
        self.show_rbc = self.cb_rbc.isChecked()
        self.show_wbc = self.cb_wbc.isChecked()
        self.show_plate = self.cb_plate.isChecked()

        self.detect_running = True
        self.status_label.setText("检测中...")
        self.btn_detect.setEnabled(False)
        self.btn_detect.setText("检测中...")

        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.current_img_paths))
        self.progress_bar.setValue(0)
        self.progress_label.setVisible(True)

        self.thread = DetectThread(
            self.model, self.current_img_paths, conf, iou,
            self.show_rbc, self.show_wbc, self.show_plate
        )
        self.thread.progress_signal.connect(self._on_progress)
        self.thread.finished_signal.connect(self.detect_finished)
        self.thread.error_signal.connect(self.detect_error)
        self.thread.finished.connect(self._on_thread_done)
        self.thread.start()

    def _on_progress(self, current, total):
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"{current} / {total}")

    def _on_thread_done(self):
        self.detect_running = False
        self.btn_detect.setEnabled(True)
        self.btn_detect.setText("开 始 检 测")
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)

    def detect_finished(self, results):
        self.current_results = results
        self.current_idx = 0
        if results:
            _, img, boxes = results[0]
            self.display_img(img)
            self.update_stat_label(boxes)
            self.update_page_label()
            total_boxes = sum(len(r[2]) for r in results)
            self.status_label.setText(f"检测完成！共检测到 {total_boxes} 个目标")
            self.img_count_label.setText(f"{len(results)} 张图片")
        else:
            self.status_label.setText("未检测到任何结果")

    def detect_error(self, msg):
        QMessageBox.critical(self, "检测错误", f"检测过程中发生错误：\n{msg}")
        self.status_label.setText("检测失败")

    def save_results(self):
        if not self.current_results:
            QMessageBox.warning(self, "提示", "暂无检测结果可保存！")
            return

        save_dir = QFileDialog.getExistingDirectory(self, "选择保存文件夹")
        if not save_dir:
            return

        save_img_dir = os.path.join(save_dir, "检测结果图")
        save_txt_dir = os.path.join(save_dir, "检测结果文件")
        os.makedirs(save_img_dir, exist_ok=True)
        os.makedirs(save_txt_dir, exist_ok=True)

        for path, img, boxes in self.current_results:
            name = os.path.splitext(os.path.basename(path))[0]
            save_img_path = os.path.join(save_img_dir, f"{name}_detect.jpg")
            cv2.imwrite(save_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            save_txt_path = os.path.join(save_txt_dir, f"{name}_result.txt")
            with open(save_txt_path, "w", encoding="utf-8") as f:
                f.write(f"文件名：{name}\n")
                f.write(f"检测目标数量：{len(boxes)}\n")
                rbc = sum(1 for b in boxes if b[4] == "RBC")
                wbc = sum(1 for b in boxes if b[4] == "WBC")
                plt = sum(1 for b in boxes if b[4] == "Platelet")
                f.write(f"RBC：{rbc}，WBC：{wbc}，Platelet：{plt}\n")
                f.write("-" * 30 + "\n")
                for i, (x1, y1, x2, y2, cls, conf) in enumerate(boxes):
                    f.write(f"目标{i + 1}：{cls}，置信度：{conf:.3f}，坐标：({x1},{y1})-({x2},{y2})\n")

        QMessageBox.information(self, "保存成功", f"结果已保存至：\n{save_dir}")
        self.status_label.setText("结果已保存")

    def clear_display(self):
        self.current_img_paths = []
        self.current_results = []
        self.current_idx = 0
        self._current_img_copy = None
        self._current_qt_img = None
        self.img_label.setText("\n\n\n请导入图片进行检测\n\n支持 JPG / PNG / BMP / TIFF 格式\n\n\n")
        self.img_label.setStyleSheet(
            "color: #a0aec0; font-size: 14px; border: 2px dashed #e2e8f0; "
            "border-radius: 12px; background-color: #fafbfc;"
        )
        self.status_label.setText("就绪")
        self.img_count_label.setText("")
        self.stat_label.setText("暂无数据")
        self.update_page_label()
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    window = RBCDetectWindow()
    window.showMaximized()
    sys.exit(app.exec_())
