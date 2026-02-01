import sys
import os
import platform
import shutil
import traceback
import time

# ==============================================================================
# 🛡️ Ultra 9 专属保命补丁 (必须放在所有 import 之前)
# ==============================================================================

# 1. 强制使用 AVX2 指令集 (避开 Ultra 9 的 AVX-VNNI/AMX 潜在 Bug)
os.environ["MKL_ENABLE_INSTRUCTIONS"] = "AVX2"

# 2. 限制单线程启动 (防止 Intel 大小核调度导致死锁崩溃)
# 等模型加载完，我们在内部再放开，但启动时必须稳！
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 3. 强制国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

print("***************************************************")
print("        正在启动 (Ultra 9 兼容模式)               ")
print("***************************************************")

from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QTextEdit, QProgressBar, QMessageBox, QFileDialog, 
                             QFrame, QGridLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QRectF
from PyQt6.QtGui import QFont, QColor, QPainter, QPainterPath

# 引入 ctypes 用于弹窗
import ctypes

try:
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

# === 全局配置 ===
IS_MAC = (platform.system() == 'Darwin')
UI_FONT = "Microsoft YaHei" if not IS_MAC else "PingFang SC"

MODEL_MAP = {
    "medium":   "systran/faster-whisper-medium",
    "base":     "systran/faster-whisper-base",
    "large-v3": "systran/faster-whisper-large-v3",
    "small":    "systran/faster-whisper-small"
}
MODEL_EXPECTED_SIZE = {
    "medium": 1500, "base": 145, "large-v3": 3050, "small": 480
}
MODEL_OPTIONS = [
    {"name": "🌟 推荐模式", "desc": "精准与速度平衡", "code": "medium", "color": "#2ecc71"},
    {"name": "🚀 极速模式", "desc": "速度最快", "code": "base", "color": "#3498db"},
    {"name": "🧠 深度模式", "desc": "超准 but 稍慢", "code": "large-v3", "color": "#00cec9"},
    {"name": "⚡ 省电模式", "desc": "轻量级", "code": "small", "color": "#1abc9c"}
]

# === 自定义按钮 ===
class ProgressButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self._progress = 0.0
        self._is_processing = False
        self.default_text = text
        self.format_str = "运行中 {0}%" 
        self._custom_text = None 
        self.setStyleSheet("""
            QPushButton { background-color: #0078d7; color: white; border-radius: 30px; font-weight: bold; font-size: 20px; }
            QPushButton:hover { background-color: #0063b1; }
            QPushButton:pressed { background-color: #005a9e; }
            QPushButton:disabled { background-color: #cccccc; color: #888; }
        """)
    def set_progress(self, value):
        if value > self._progress: self._progress = float(value)
        self.update() 
    def set_text_override(self, text):
        self._custom_text = text
        self.update()
    def set_format(self, fmt):
        self.format_str = fmt
        self._custom_text = None
        self.update()
    def start_processing(self):
        self._is_processing = True
        self._progress = 0.0
        self._custom_text = None
        self.format_str = "准备中 {0}%"
        self.setEnabled(False) 
        self.update()
    def stop_processing(self):
        self._is_processing = False
        self._progress = 0.0
        self._custom_text = None
        self.setText(self.default_text)
        self.setEnabled(True)
        self.update()
    def paintEvent(self, event):
        if not self._is_processing:
            super().paintEvent(event)
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        rectf = QRectF(rect)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#f0f0f0"))
        painter.drawRoundedRect(rectf, 30, 30)
        if self._progress > 0:
            prog_width = (rect.width() * (self._progress / 100.0))
            if prog_width < 30: prog_width = 30
            path = QPainterPath()
            path.addRoundedRect(rectf, 30, 30)
            painter.setClipPath(path)
            painter.setBrush(QColor("#0078d7"))
            painter.drawRect(0, 0, int(prog_width), int(rect.height()))
            painter.setClipping(False)
        painter.setPen(QColor("#333") if self._progress < 55 else QColor("white"))
        font = self.font()
        font.setPointSize(16) 
        painter.setFont(font)
        if self._custom_text: display_text = self._custom_text
        else: display_text = self.format_str.format(int(self._progress))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, display_text)

class DownloadMonitor(QThread):
    progress_update = pyqtSignal(int, int, int)
    def __init__(self, target_folder, expected_size_mb):
        super().__init__()
        self.target_folder = target_folder
        self.expected_size_mb = expected_size_mb
        self.is_running = True
    def get_folder_size_mb(self):
        total = 0
        try:
            for dp, dn, fn in os.walk(self.target_folder):
                for f in fn:
                    if not f.endswith(".lock"): total += os.path.getsize(os.path.join(dp, f))
        except: pass
        return int(total / (1024*1024))
    def run(self):
        while self.is_running:
            current = self.get_folder_size_mb()
            pct = 0
            if self.expected_size_mb > 0:
                pct = int((current / self.expected_size_mb) * 39)
                if pct > 39: pct = 39
            self.progress_update.emit(current, self.expected_size_mb, pct)
            time.sleep(0.5)
    def stop(self): self.is_running = False

class WorkThread(QThread):
    status_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    stage_signal = pyqtSignal(str)
    result_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    monitor_signal = pyqtSignal(bool, str, int)

    def __init__(self, video_path, model_code):
        super().__init__()
        self.video_path = video_path
        self.model_code = model_code
        self.repo_id = MODEL_MAP[model_code]
        self.is_running = True

    def run(self):
        print(f"--- 任务启动 ---")
        try:
            if getattr(sys, 'frozen', False):
                base_dir = os.path.dirname(sys.executable)
            else:
                base_dir = os.path.dirname(os.path.abspath(__file__))
            
            models_root = os.path.join(base_dir, "models")
            model_dir = os.path.join(models_root, f"models--{self.repo_id.replace('/', '--')}")

            # --- 阶段 1: 下载 ---
            self.status_signal.emit(f"⏳ 正在校验模型...")
            expected_mb = MODEL_EXPECTED_SIZE.get(self.model_code, 1000)
            self.monitor_signal.emit(True, model_dir, expected_mb)

            try:
                print("Snapshot download check...")
                snapshot_download(
                    repo_id=self.repo_id,
                    repo_type="model",
                    local_dir=model_dir,
                    resume_download=True,
                    max_workers=1
                )
            except Exception as dl_err:
                print(f"Download Error: {dl_err}")
                self.monitor_signal.emit(False, "", 0)
                if os.path.exists(model_dir):
                    self.status_signal.emit("⚠️ 尝试使用本地缓存...")
                else:
                    raise Exception(f"网络错误: {str(dl_err)}")

            self.monitor_signal.emit(False, "", 0)
            if not self.is_running: return
            self.stage_signal.emit("加载中 {0}%") 
            self.progress_signal.emit(40)

            # --- 阶段 2: 加载 (Crash Zone) ---
            self.status_signal.emit("🧠 正在唤醒 AI 引擎...")
            print(f"Loading WhisperModel from {model_dir}")
            
            try:
                # 🔥 终极兼容配置 🔥
                # 1. 不指定 device (自动)
                # 2. compute_type="float32" (最稳)
                # 3. cpu_threads=4 (手动限制线程，防止 Ultra 9 跑满崩溃)
                model = WhisperModel(
                    model_dir, 
                    device="cpu", 
                    compute_type="float32",
                    cpu_threads=4, 
                    local_files_only=True 
                )
                print("Model Loaded Successfully!")
            except Exception as e:
                print(f"CRASH: {e}")
                traceback.print_exc()
                raise e

            if not self.is_running: return
            self.stage_signal.emit("识别中 {0}%")
            self.progress_signal.emit(50)

            # --- 阶段 3: 识别 ---
            self.status_signal.emit("🎧 正在分析...")
            segments, info = model.transcribe(
                self.video_path, beam_size=5, language="zh",
                initial_prompt="这是一段清晰的普通话，请加标点符号。"
            )
            
            full_text = ""
            total_duration = info.duration
            print(f"Duration: {total_duration}")

            for segment in segments:
                if not self.is_running: return
                full_text += segment.text
                print(f"Seg: {segment.text}")
                if total_duration > 0:
                    progress = 50 + int((segment.end / total_duration) * 48)
                    self.progress_signal.emit(progress)

            self.progress_signal.emit(100)
            self.status_signal.emit("✅ 转换完成！")
            self.result_signal.emit(full_text)

        except Exception as e:
            print(f"Exception: {e}")
            traceback.print_exc()
            self.monitor_signal.emit(False, "", 0)
            self.error_signal.emit(str(e))

    def stop(self): self.is_running = False

class ModelCard(QPushButton):
    def __init__(self, title, desc, code, color, parent=None):
        super().__init__(parent)
        self.code = code
        self.default_color = color
        self.setCheckable(True)
        self.setFixedHeight(100) 
        layout = QVBoxLayout(self)
        self.lbl_title = QLabel(title)
        self.lbl_title.setFont(QFont(UI_FONT, 15, QFont.Weight.Bold))
        self.lbl_desc = QLabel(desc)
        self.lbl_desc.setFont(QFont(UI_FONT, 13))
        layout.addWidget(self.lbl_title)
        layout.addWidget(self.lbl_desc)
        self.update_style(False)
    def update_style(self, selected):
        if selected:
            self.setStyleSheet(f"QPushButton {{ background-color: {self.default_color}15; border: 3px solid {self.default_color}; border-radius: 12px; }}")
            self.lbl_title.setStyleSheet(f"color: {self.default_color};")
        else:
            self.setStyleSheet("QPushButton { background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 12px; }")
            self.lbl_title.setStyleSheet("color: #333;")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("❤️ 专属语音转文字助手 (兼容修复版)")
        self.resize(1100, 700) 
        self.setAcceptDrops(True)
        self.video_path = ""
        self.selected_model = "medium"
        self.worker = None
        self.monitor = None
        self.model_btns = []
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        left_widget = QWidget(); left_layout = QVBoxLayout(left_widget)
        
        self.import_area = QPushButton("\n📂 点击上传 / 拖拽视频\n(再次点击可替换)\n")
        self.import_area.setFixedHeight(140)
        self.import_area.clicked.connect(self.select_video)
        left_layout.addWidget(self.import_area)

        model_layout = QGridLayout()
        for i, m in enumerate(MODEL_OPTIONS):
            btn = ModelCard(m["name"], m["desc"], m["code"], m["color"])
            btn.clicked.connect(lambda checked, b=btn: self.on_model_click(b))
            model_layout.addWidget(btn, i // 2, i % 2)
            self.model_btns.append(btn)
        left_layout.addLayout(model_layout)
        self.on_model_click(self.model_btns[0])
        
        self.lbl_status = QLabel("准备就绪")
        left_layout.addWidget(self.lbl_status)
        
        self.btn_start = ProgressButton("开始转换")
        self.btn_start.setFixedHeight(60)
        self.btn_start.setEnabled(False) 
        self.btn_start.clicked.connect(self.start_process)
        left_layout.addWidget(self.btn_start)

        right_widget = QWidget(); right_layout = QVBoxLayout(right_widget)
        self.text_area = QTextEdit()
        right_layout.addWidget(self.text_area)
        self.btn_copy = QPushButton("📋 复制")
        self.btn_copy.clicked.connect(self.copy_text)
        right_layout.addWidget(self.btn_copy)
        
        main_layout.addWidget(left_widget, 4); main_layout.addWidget(right_widget, 6)
        self.setLayout(main_layout)

    def on_model_click(self, clicked_btn):
        for btn in self.model_btns:
            is_target = (btn == clicked_btn)
            btn.setChecked(is_target)
            btn.update_style(is_target)
        self.selected_model = clicked_btn.code
    def dragEnterEvent(self, e): e.accept() if e.mimeData().hasUrls() else e.ignore()
    def dropEvent(self, e): self.load_video(e.mimeData().urls()[0].toLocalFile())
    def select_video(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "Media (*.mp4 *.mov *.avi)")
        if f: self.load_video(f)
    def load_video(self, path):
        self.video_path = path
        self.import_area.setText(f"已加载: {os.path.basename(path)}")
        self.btn_start.setEnabled(True)
    def start_process(self):
        if not self.video_path: return
        self.import_area.setEnabled(False)
        self.btn_start.start_processing()
        self.worker = WorkThread(self.video_path, self.selected_model)
        self.worker.status_signal.connect(self.lbl_status.setText) 
        self.worker.progress_signal.connect(lambda v: self.btn_start.set_progress(v)) 
        self.worker.stage_signal.connect(lambda s: self.btn_start.set_format(s))
        self.worker.monitor_signal.connect(self.handle_monitor)
        self.worker.result_signal.connect(self.on_success)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()
    def handle_monitor(self, start, path, size):
        if start:
            if self.monitor: self.monitor.stop()
            self.monitor = DownloadMonitor(path, size)
            self.monitor.progress_update.connect(lambda c,t,p: self.btn_start.set_text_override(f"下载中 {c}MB/{t}MB"))
            self.monitor.start()
        else:
            if self.monitor: self.monitor.stop()
    def on_success(self, text):
        self.btn_start.set_progress(100)
        self.text_area.setPlainText(text)
        self.reset_ui()
    def on_error(self, msg):
        self.reset_ui()
        self.lbl_status.setText("❌ 出错")
        QMessageBox.warning(self, "错误", f"发生错误: {msg}\n\n请查看黑框框！")
    def reset_ui(self):
        self.btn_start.stop_processing()
        self.import_area.setEnabled(True)
    def copy_text(self):
        self.text_area.selectAll()
        self.text_area.copy()

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        traceback.print_exc()
        input("Press Enter...")