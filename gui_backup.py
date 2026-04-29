import os
import sys

import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ultralytics import YOLO


class DetectThread(QThread):
    finished_signal = pyqtSignal(list)
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

        self.class_map = {0: "RBC", 1: "WBC", 2: "Platelet"}

    def run(self):
        try:
            results = []
            for path in self.img_paths:
                pred = self.model.predict(
                    source=path, conf=self.conf_thres, iou=self.iou_thres, save=False, verbose=False
                )[0]

                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                filtered_boxes = []
                for box in pred.boxes:
                    cls_id = int(box.cls)
                    cls_name = self.class_map.get(cls_id, "unknown")

                    if (
                        (cls_name == "RBC" and self.show_rbc)
                        or (cls_name == "WBC" and self.show_wbc)
                        or (cls_name == "Platelet" and self.show_plate)
                    ):
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf)
                        filtered_boxes.append([x1, y1, x2, y2, cls_name, conf])

                for x1, y1, x2, y2, cls_name, conf in filtered_boxes:
                    if cls_name == "RBC":
                        color = (255, 0, 0)
                    elif cls_name == "WBC":
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)

                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, f"{cls_name} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                results.append((path, img, filtered_boxes))
            self.finished_signal.emit(results)
        except Exception as e:
            self.error_signal.emit(str(e))


class RBCDetectWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("血细胞检测系统")
        self.setGeometry(100, 100, 1200, 700)

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

    def load_model(self, path):
        try:
            self.model = YOLO(path)
            self.model_path = path
            self.model_label.setText(f"模型：{os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败：{e!s}")

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ========== 左侧控制面板 ==========
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setSpacing(10)
        control_widget.setMaximumWidth(280)

        # 0. 模型选择
        model_group = QGroupBox("模型")
        model_layout = QVBoxLayout(model_group)
        self.model_label = QLabel("未加载模型")
        self.model_label.setWordWrap(True)
        self.btn_load_model = QPushButton("加载模型")
        self.btn_load_model.clicked.connect(self.select_model)
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.btn_load_model)
        control_layout.addWidget(model_group)

        # 尝试自动加载默认模型
        default_model = "runs/detect/train7/weights/best.pt"
        if os.path.exists(default_model):
            self.load_model(default_model)
        else:
            self.model_label.setText("模型：未找到默认模型，请手动加载")

        # 1. 类别显示控制
        class_group = QGroupBox("类别显示")
        class_layout = QVBoxLayout(class_group)
        self.cb_rbc = QCheckBox("显示红细胞(RBC)")
        self.cb_wbc = QCheckBox("显示白细胞(WBC)")
        self.cb_plate = QCheckBox("显示血小板(Platelet)")
        self.cb_rbc.setChecked(True)
        self.cb_wbc.setChecked(True)
        self.cb_plate.setChecked(True)
        class_layout.addWidget(self.cb_rbc)
        class_layout.addWidget(self.cb_wbc)
        class_layout.addWidget(self.cb_plate)
        control_layout.addWidget(class_group)

        # 2. 阈值控制
        conf_group = QGroupBox("置信度阈值")
        conf_layout = QHBoxLayout(conf_group)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(25)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setSingleStep(0.01)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_spin)
        control_layout.addWidget(conf_group)

        iou_group = QGroupBox("IoU阈值")
        iou_layout = QHBoxLayout(iou_group)
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(1, 100)
        self.iou_slider.setValue(45)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setValue(0.45)
        self.iou_spin.setSingleStep(0.01)
        iou_layout.addWidget(self.iou_slider)
        iou_layout.addWidget(self.iou_spin)
        control_layout.addWidget(iou_group)

        # 3. 图片导入
        img_group = QGroupBox("图片导入")
        img_layout = QVBoxLayout(img_group)
        self.btn_single = QPushButton("导入单张图片")
        self.btn_folder = QPushButton("导入文件夹")
        img_layout.addWidget(self.btn_single)
        img_layout.addWidget(self.btn_folder)
        control_layout.addWidget(img_group)

        # 4. 操作按钮
        self.btn_detect = QPushButton("开始检测")
        self.btn_save = QPushButton("保存检测结果")
        self.btn_clear = QPushButton("清空显示")
        control_layout.addWidget(self.btn_detect)
        control_layout.addWidget(self.btn_save)
        control_layout.addWidget(self.btn_clear)

        # 5. 图片导航
        nav_group = QGroupBox("图片导航")
        nav_layout = QHBoxLayout(nav_group)
        self.btn_prev = QPushButton("上一张")
        self.btn_next = QPushButton("下一张")
        self.page_label = QLabel("0/0")
        self.page_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.page_label)
        nav_layout.addWidget(self.btn_next)
        control_layout.addWidget(nav_group)

        # 6. 统计信息
        stat_group = QGroupBox("检测统计")
        stat_layout = QVBoxLayout(stat_group)
        self.stat_label = QLabel("暂无数据")
        self.stat_label.setWordWrap(True)
        stat_layout.addWidget(self.stat_label)
        control_layout.addWidget(stat_group)

        control_layout.addStretch()

        # ========== 右侧显示区域 ==========
        display_frame = QFrame()
        display_frame.setFrameShape(QFrame.Box)
        display_layout = QVBoxLayout(display_frame)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setAlignment(Qt.AlignCenter)
        self.img_label = QLabel("请导入图片进行检测\n\n支持 JPG/PNG/BMP 格式")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.img_label.setMinimumSize(400, 300)
        self.scroll.setWidget(self.img_label)
        display_layout.addWidget(self.scroll)

        self.status_label = QLabel("就绪")
        display_layout.addWidget(self.status_label)

        main_layout.addWidget(control_widget, stretch=1)
        main_layout.addWidget(display_frame, stretch=4)

        # ========== 信号绑定 ==========
        # 滑块与数值框联动（阻断信号防止无限递归）
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

    # ========== 滑块联动（无递归） ==========
    def _on_conf_slider_changed(self, v):
        self.conf_spin.blockSignals(True)
        self.conf_spin.setValue(v / 100)
        self.conf_spin.blockSignals(False)

    def _on_conf_spin_changed(self, v):
        self.conf_slider.blockSignals(True)
        self.conf_slider.setValue(int(v * 100))
        self.conf_slider.blockSignals(False)

    def _on_iou_slider_changed(self, v):
        self.iou_spin.blockSignals(True)
        self.iou_spin.setValue(v / 100)
        self.iou_spin.blockSignals(False)

    def _on_iou_spin_changed(self, v):
        self.iou_slider.blockSignals(True)
        self.iou_slider.setValue(int(v * 100))
        self.iou_slider.blockSignals(False)

    # ========== 功能函数 ==========
    def select_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "PyTorch模型 (*.pt *.pth);;所有文件 (*.*)")
        if path:
            self.load_model(path)

    def load_single_img(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)")
        if path:
            self.current_img_paths = [path]
            self.current_idx = 0
            self.show_image_at_idx(0)
            self.status_label.setText(f"已导入：{os.path.basename(path)}")

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
            QMessageBox.warning(self, "警告", "所选文件夹中没有图片文件！")
            return
        self.current_img_paths = paths
        self.current_idx = 0
        self.show_image_at_idx(0)
        self.status_label.setText(f"已导入文件夹，共 {len(paths)} 张图片")

    def show_image_at_idx(self, idx):
        if 0 <= idx < len(self.current_img_paths):
            img = cv2.imread(self.current_img_paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.display_img(img)
            self.update_page_label()

    def update_page_label(self):
        total = len(self.current_img_paths)
        if total > 0:
            cur = self.current_idx + 1 if self.current_idx < total else 0
            self.page_label.setText(f"{cur}/{total}")
        else:
            self.page_label.setText("0/0")

    def prev_image(self):
        if self.current_results:
            total = len(self.current_results)
            if total == 0:
                return
            self.current_idx = (self.current_idx - 1) % total
            _, img, boxes = self.current_results[self.current_idx]
            self.display_img(img)
            self.update_stat_label(boxes)
            self.update_page_label()
        elif self.current_img_paths:
            total = len(self.current_img_paths)
            if total == 0:
                return
            self.current_idx = (self.current_idx - 1) % total
            self.show_image_at_idx(self.current_idx)

    def next_image(self):
        if self.current_results:
            total = len(self.current_results)
            if total == 0:
                return
            self.current_idx = (self.current_idx + 1) % total
            _, img, boxes = self.current_results[self.current_idx]
            self.display_img(img)
            self.update_stat_label(boxes)
            self.update_page_label()
        elif self.current_img_paths:
            total = len(self.current_img_paths)
            if total == 0:
                return
            self.current_idx = (self.current_idx + 1) % total
            self.show_image_at_idx(self.current_idx)

    def display_img(self, img):
        if img is None:
            return
        h, w, ch = img.shape
        bytes_per_line = ch * w
        # 复制数据防止 numpy 数组被 GC 导致悬空指针
        img_copy = img.copy()
        qt_img = QImage(img_copy.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        # 缩放到 label 大小以支持窗口缩放
        scaled = pixmap.scaled(self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_label.setPixmap(scaled)
        # 持有引用防止被回收
        self._current_qt_img = qt_img
        self._current_img_copy = img_copy

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 窗口缩放时重新调整图片显示
        if hasattr(self, "_current_img_copy") and self._current_img_copy is not None:
            self.display_img(self._current_img_copy)

    def update_stat_label(self, boxes):
        rbc_count = sum(1 for b in boxes if b[4] == "RBC")
        wbc_count = sum(1 for b in boxes if b[4] == "WBC")
        plt_count = sum(1 for b in boxes if b[4] == "Platelet")
        self.stat_label.setText(
            f"RBC（红细胞）：{rbc_count}\n"
            f"WBC（白细胞）：{wbc_count}\n"
            f"Platelet（血小板）：{plt_count}\n"
            f"总计：{len(boxes)}"
        )

    def start_detect(self):
        if not self.model:
            QMessageBox.warning(self, "警告", "模型未加载！请先加载模型。")
            return
        if not self.current_img_paths:
            QMessageBox.warning(self, "警告", "请先导入图片！")
            return
        if self.detect_running:
            QMessageBox.warning(self, "警告", "检测正在进行中，请稍候...")
            return

        conf = self.conf_spin.value()
        iou = self.iou_spin.value()
        self.show_rbc = self.cb_rbc.isChecked()
        self.show_wbc = self.cb_wbc.isChecked()
        self.show_plate = self.cb_plate.isChecked()

        self.detect_running = True
        self.status_label.setText("检测中...")
        self.btn_detect.setEnabled(False)

        self.thread = DetectThread(
            self.model, self.current_img_paths, conf, iou, self.show_rbc, self.show_wbc, self.show_plate
        )
        self.thread.finished_signal.connect(self.detect_finished)
        self.thread.error_signal.connect(self.detect_error)
        self.thread.finished.connect(self._on_thread_done)
        self.thread.start()

    def _on_thread_done(self):
        self.detect_running = False
        self.btn_detect.setEnabled(True)

    def detect_finished(self, results):
        self.current_results = results
        self.current_idx = 0
        if results:
            _, img, boxes = results[0]
            self.display_img(img)
            self.update_stat_label(boxes)
            self.update_page_label()
            self.status_label.setText(f"检测完成！共 {len(results)} 张图片")
        else:
            self.status_label.setText("未检测到任何结果")

    def detect_error(self, msg):
        QMessageBox.critical(self, "检测错误", msg)
        self.status_label.setText("检测失败")

    def save_results(self):
        if not self.current_results:
            QMessageBox.warning(self, "警告", "暂无检测结果可保存！")
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

        QMessageBox.information(self, "成功", f"结果已保存至：\n{save_dir}")
        self.status_label.setText("结果已保存")

    def clear_display(self):
        self.current_img_paths = []
        self.current_results = []
        self.current_idx = 0
        self._current_img_copy = None
        self._current_qt_img = None
        self.img_label.setText("请导入图片进行检测\n\n支持 JPG/PNG/BMP 格式")
        self.status_label.setText("就绪")
        self.stat_label.setText("暂无数据")
        self.update_page_label()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RBCDetectWindow()
    window.show()
    sys.exit(app.exec_())
