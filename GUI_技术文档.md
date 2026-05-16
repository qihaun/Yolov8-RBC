# 血细胞自动检测系统 — GUI 技术文档

## 一、系统概述

基于 PyQt5 框架开发的桌面端图形化血细胞检测应用。集成 YOLOv8 目标检测模型，支持血液涂片图像中红细胞(RBC)、白细胞(WBC)、血小板(Platelet)的自动检测、可视化标注和结果导出。

| 项目 | 说明 |
|---|---|
| 文件 | `gui.py` |
| 框架 | PyQt5 (Qt5) |
| 模型 | Ultralytics YOLOv8 (`best.pt`) |
| 图像处理 | OpenCV (`cv2`) |
| 运行环境 | Windows + Anaconda (`yolov8biye`) |

---

## 二、架构设计

### 2.1 类结构

```
QThread
  └── DetectThread    # 后台检测线程

QMainWindow
  └── RBCDetectWindow # 主界面窗口
```

### 2.2 线程模型

```
┌─────────────┐     progress_signal(int,int)     ┌──────────────┐
│DetectThread │ ──┬────────────────────────────── │ update UI    │
│ (子线程)     │   │ finished_signal(list)        │ (主线程)      │
│ run()       │ ──┼────────────────────────────── │ slots        │
│ ...         │   │ error_signal(str)             │              │
│             │ ──┼────────────────────────────── │              │
│             │   │ QThread.finished              │              │
│             │ ──┴────────────────────────────── │              │
└─────────────┘                                   └──────────────┘
```

**为什么用多线程：** YOLO 推理是 CPU/GPU 密集型操作, 若在主线程执行会阻塞 Qt 事件循环, 导致界面卡死。用 `QThread` 将推理放到后台, 主线程持续处理 UI 事件, 进度条和按钮响应保持流畅。

**信号-槽连接注意事项:**
- 跨线程连接自动使用 `Qt.QueuedConnection`, 信号排队到主线程事件队列处理
- 每次 `start_detect()` 前先 `disconnect()` 旧线程信号, 防止重复连接导致信号累积

### 2.3 数据流

```
用户点击"开始检测"
  │
  ├─ 前置检查 (模型? 图片? 检测中?)
  │
  ├─ 创建 DetectThread(当前参数) → 启动
  │
  ├─ 子线程逐张:
  │    model.predict(path) → 筛选检测框 → cv2 绘图 → 发射进度
  │
  ├─ 全部完成后:
  │    finished_signal.emit(results) → detect_finished()
  │    QThread.finished → _on_thread_done()
  │
  └─ 主界面:
       current_results = results
       显示第一张结果图 + 统计信息
```

---

## 三、界面布局

### 3.1 整体结构

```
QMainWindow
  └── QWidget (central_widget)
        └── QHBoxLayout (main_layout)
              ├── QScrollArea (left_scroll)           ← 左侧控制面板
              │     └── QWidget (control_widget)
              │           └── QVBoxLayout
              │                 ├── QFrame (标题栏 + 状态指示)
              │                 ├── QGroupBox (模型管理)
              │                 ├── QGroupBox (类别显示控制)
              │                 ├── QGroupBox (阈值设置)
              │                 ├── QGroupBox (图片导入)
              │                 ├── QGroupBox (操作)
              │                 ├── QProgressBar + QLabel (进度)
              │                 ├── QGroupBox (图片导航)
              │                 ├── QGroupBox (检测统计)
              │                 └── Stretch
              │
              └── QFrame (display_panel)              ← 右侧显示区域
                    └── QVBoxLayout
                          ├── QScrollArea (图像查看器)
                          │     └── QLabel (img_label)
                          └── QFrame (底部状态栏)
                                ├── status_label
                                └── img_count_label
```

### 3.2 控件清单

| 控件 | 对象名 | 类型 | 用途 |
|---|---|---|---|
| 标题 | `title` | QLabel | 显示"血细胞检测系统" |
| 状态指示 | `status_indicator` | QLabel | 显示模型加载状态 |
| 模型标签 | `model_label` | QLabel | 显示已加载的模型文件名 |
| 加载模型 | `btn_load_model` | QPushButton | 手动选择 .pt 模型文件 |
| RBC 复选框 | `cb_rbc` | QCheckBox | 显示/隐藏红细胞检测框 |
| WBC 复选框 | `cb_wbc` | QCheckBox | 显示/隐藏白细胞检测框 |
| PLT 复选框 | `cb_plate` | QCheckBox | 显示/隐藏血小板检测框 |
| 置信度滑块 | `conf_slider` | QSlider | 调节置信度阈值 (1-100) |
| 置信度数值 | `conf_spin` | QDoubleSpinBox | 显示/输入置信度 (0.01-1.00) |
| IoU 滑块 | `iou_slider` | QSlider | 调节 IoU 阈值 (1-100) |
| IoU 数值 | `iou_spin` | QDoubleSpinBox | 显示/输入 IoU (0.01-1.00) |
| 导入单张 | `btn_single` | QPushButton | 选择单张图片文件 |
| 导入文件夹 | `btn_folder` | QPushButton | 选择图片文件夹 |
| 开始检测 | `btn_detect` | QPushButton | 启动检测任务 |
| 保存结果 | `btn_save` | QPushButton | 导出检测结果 |
| 清空显示 | `btn_clear` | QPushButton | 清空所有数据和显示 |
| 进度条 | `progress_bar` | QProgressBar | 显示检测进度 |
| 进度文字 | `progress_label` | QLabel | 显示"3/15"进度文本 |
| 上一张 | `btn_prev` | QPushButton | 切换到上一张图片 |
| 下一张 | `btn_next` | QPushButton | 切换到下一张图片 |
| 页码 | `page_label` | QLabel | 显示"2/10"页码 |
| 统计 | `stat_label` | QLabel | 显示各类别计数 |
| 图像显示 | `img_label` | QLabel | 显示检测结果图像 |
| 状态栏 | `status_label` | QLabel | 底部操作状态文字 |
| 图片计数 | `img_count_label` | QLabel | 底部"10 张图片"计数 |

---

## 四、关键技术实现

### 4.1 模型加载

```python
def load_model(self, path):
    self.model = YOLO(path)  # 仅将 YOLO() 放在 try 中
    # UI 更新放在 try 外, 避免非致命 UI 错误误报"模型加载失败"
    self.model_label.setText(os.path.basename(path))
    self.status_indicator.setText("模型已就绪")
```

**设计要点:** 只把模型加载本身放在 `try/except` 中。如果 YOLO 加载成功但 `setStyleSheet` 等 UI 操作抛异常, 不会误弹"模型加载失败"对话框。

### 4.2 滑块-数值框联动 (无递归)

```python
def _on_slider_changed(self, v):
    self.spin.blockSignals(True)   # 阻断接收方信号
    self.spin.setValue(v / 100)
    self.spin.blockSignals(False)  # 恢复
```

**为什么用 `blockSignals`:** 滑块和数值框的 `valueChanged` 信号互相连接, 若不用 `blockSignals` 阻断, 会形成无限递归: 滑块变→更新数值框→数值框发射信号→更新滑块→滑块发射信号→...

### 4.3 numpy 图像 → QImage 转换

```python
def display_img(self, img):
    img_copy = img.copy()  # 深拷贝, 防止 numpy 原始数据被 GC
    qt_img = QImage(img_copy.data, w, h, bytes_per_line, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qt_img)
    scaled = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
    label.setPixmap(scaled)
    self._current_qt_img = qt_img       # 持有引用, 防止被回收
    self._current_img_copy = img_copy   # 持有引用
```

**悬空指针问题:** `QImage(img.data, ...)` 直接引用 numpy 数组的内存缓冲区。如果 numpy 数组被 Python GC 回收, QImage 内部指针指向已释放内存, 显示时可能崩溃或花屏。解决: 用 `.copy()` 创建独立副本, 并通过实例变量持有引用。

### 4.4 窗口缩放自适应

```python
def resizeEvent(self, event):
    super().resizeEvent(event)
    if self._current_img_copy is not None:
        self.display_img(self._current_img_copy)  # 重新缩放
```

每次窗口大小改变时, 用缓存的图像数据重新计算缩放尺寸, 保证图像始终填满显示区域。

### 4.5 多线程信号管理

```python
# 新检测前先断开旧连接
if hasattr(self, 'thread') and self.thread is not None:
    self.thread.progress_signal.disconnect()
    self.thread.finished_signal.disconnect()
    ...
```

**信号累积 Bug 修复:** 每次 `start_detect()` 创建新线程, 旧线程的信号连接若不断开, 多次检测后所有旧线程的信号都会触发当前回调, 导致 `current_results` 被旧数据覆盖。

### 4.6 图片导航与页码一致性

```python
def _get_total(self):
    return len(self.current_results) if self.current_results else len(self.current_img_paths)
```

导航按钮和页码显示都通过 `_get_total()` 获取图片总数, 保证一致性: 有检测结果时用结果数量, 未检测时用导入图片数量。

---

## 五、启动流程

```
main.py / gui.py
  ├── QApplication 初始化
  ├── RBCDetectWindow.__init__()
  │     ├── 设置窗口属性 (标题/大小/最小尺寸)
  │     ├── 初始化状态变量 (paths/results/idx/flags)
  │     ├── init_ui()  ← 构建所有控件和布局
  │     │     ├── 创建左侧控制面板 (各组控件)
  │     │     ├── 创建右侧显示区域
  │     │     ├── 绑定信号-槽连接
  │     │     └── 尝试自动加载默认模型
  │     │           └── load_model(".../best.pt")
  │     └── apply_theme()  ← 应用全局 QSS 样式表
  │
  └── window.showMaximized()  ← 全屏显示
        └── app.exec_()         ← 进入 Qt 事件循环
```

---

## 六、文件依赖

| 依赖 | 说明 |
|---|---|
| `ultralytics` | YOLO 模型加载与推理 |
| `PyQt5.QtCore` | Qt 核心类 (QThread, pyqtSignal, Qt 常量) |
| `PyQt5.QtWidgets` | UI 控件 (按钮/标签/滑块/布局...) |
| `PyQt5.QtGui` | 图像/字体类 (QImage, QPixmap, QFont) |
| `cv2` (opencv-python) | 图像读取、颜色空间转换、检测框绘制 |
| `numpy` | 数组操作 |
| `os` | 文件路径处理 |
| `sys` | 程序退出 |
| `traceback` | 异常调用栈打印 |

---

## 七、已知问题与注意事项

1. **GPU 内存限制** — RTX 3050 4GB 显存, `batch=8, imgsz=800` 已接近上限, 不要同时运行多个模型
2. **torch < 2.0 警告** — 当前环境 PyTorch 1.13.1, 训练时有非确定性警告, 不影响推理
3. **emoji 渲染** — Windows 上按钮中的 emoji 可能导致文字模糊, 已全部移除改用纯文字
4. **路径处理** — 默认模型路径使用 `os.path.join(__file__)` 构建绝对路径, 避免工作目录变化导致找不到模型
