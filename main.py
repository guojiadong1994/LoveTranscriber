import sys
import os
import platform
import shutil
import traceback
import time
import socket
import ctypes

# ==============================================================================
# 🛡️ 0. 必须最先执行的“保命”配置 (针对 Intel Ultra 9 内存冲突)
# ==============================================================================

# 【核心修复】强制降级指令集，防止 Ultra 9 触发 Access Violation
os.environ["MKL_ENABLE_INSTRUCTIONS"] = "AVX2"
os.environ["CT2_USE_EXPERIMENTAL_PACKED_GEMM"] = "1"

# 【防死锁】初始化时强制单线程，防止 MKL 库打架
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 强制国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 即使是调试版，也建议关掉官方进度条，防止编码报错，我们自己打印进度
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

# ==============================================================================
# 🛡️ 1. 双重日志系统 (屏幕+文件)
# ==============================================================================

# 确定大本营目录
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_FILE = os.path.join(BASE_DIR, "crash.log")

class DualWriter:
    """
    这个类的作用是：
    1. 把字写在黑框框里 (给你看)
    2. 把字写在 txt 文件里 (给开发者看)
    保证一条报错都漏不掉！
    """
    def __init__(self, original_stream, log_file_path):
        self.original_stream = original_stream
        self.log_file = open(log_file_path, "w", encoding="utf-8", buffering=1) # "w" 模式每次清空旧日志

    def write(self, message):
        # 1. 屏幕输出
        try:
            if self.original_stream:
                self.original_stream.write(message)
                self.original_stream.flush()
        except: pass
        
        # 2. 文件记录
        try:
            self.log_file.write(message)
            self.log_file.flush()
        except: pass

    def flush(self):
        try:
            if self.original_stream: self.original_stream.flush()
        except: pass
        try: self.log_file.flush()
        except: pass

# 立即接管输出
sys.stdout = DualWriter(sys.stdout, LOG_FILE)
sys.stderr = DualWriter(sys.stderr, LOG_FILE)

import faulthandler
# 开启底层崩溃捕捉 (针对 Access Violation)
# 注意：faulthandler 只能绑定一个文件句柄，我们绑定到日志文件
try:
    faulthandler.enable(file=sys.stdout.log_file, all_threads=True)
except:
    print("Warning: faulthandler init failed")

print(f"===== APP START {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
print(f"System: {platform.uname()}")
print(f"Fix applied: AVX2 Enforced")

# ==============================================================================
# 🛡️ 2. 导入 UI 和 AI 库
# ==============================================================================
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QTextEdit, QProgressBar, QMessageBox, QFileDialog, 
                             QFrame, QGridLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QRectF
from PyQt6.QtGui import QFont, QColor, QPainter, QPainterPath

try:
    print("Importing faster_whisper...")
    from faster_whisper import WhisperModel
    print("Importing huggingface_hub...")
    from huggingface_hub import snapshot_download
    HAS_WHISPER = True
    print("✅ Libraries imported successfully")
except ImportError as e:
    print(f"❌ Missing libraries: {e}")
    HAS_WHISPER = False
except Exception as e:
    print(f"❌ Import error: {e}")
    traceback.print_exc()

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
    def auto_creep_progress(self):
        current = self._progress
        increment = 0.0
        if current < 39.0:
            if current < 15.0: increment = 0.5 
            elif current < 30.0: increment = 0.1 
            else: increment = 0.01 
        elif current >= 40.0 and current < 49.0:
            increment = 0.05
        elif current >= 50.0 and current < 98.0:
            increment = 0.1
        self._progress += increment
        if current < 40.0 and self._progress >= 39.9: self._progress = 39.9
        if current < 50.0 and self._progress >= 49.9: self._progress = 49.9
        if self._progress >= 99.0: self._progress = 99.0
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

# === 监控线程 (监控 BASE_DIR/models) ===
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

# === 核心工作线程 ===
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
        print(f"--- TASK START: {self.model_code} ---")
        if not HAS_WHISPER:
            self.error_signal.emit("错误：缺少 faster-whisper 库")
            return

        try:
            # 1. 确保模型目录存在 (在 BASE_DIR 下)
            models_root = os.path.join(BASE_DIR, "models")
            os.makedirs(models_root, exist_ok=True)
            
            # Huggingface cache 结构目录
            model_base_dir = os.path.join(models_root, f"models--{self.repo_id.replace('/', '--')}")
            print(f"Model Base Dir: {model_base_dir}")

            # --- 阶段 1: 下载 ---
            self.status_signal.emit(f"⏳ 正在校验/下载模型...")
            expected_mb = MODEL_EXPECTED_SIZE.get(self.model_code, 1000)
            self.monitor_signal.emit(True, model_base_dir, expected_mb)

            try:
                print("Calling snapshot_download...")
                # 使用返回值获取真实路径
                real_model_path = snapshot_download(
                    repo_id=self.repo_id,
                    repo_type="model",
                    local_dir=model_base_dir,
                    resume_download=True,
                    max_workers=1
                )
                print(f"✅ Real Snapshot Path: {real_model_path}")
            except Exception as dl_err:
                print(f"Download Error: {dl_err}")
                self.monitor_signal.emit(False, "", 0)
                # 容错：尝试使用 base_dir
                if os.path.exists(model_base_dir) and self.get_size(model_base_dir) > (expected_mb * 0.8):
                    print("Fallback to local cache...")
                    self.status_signal.emit("⚠️ 网络微恙，尝试使用本地缓存...")
                    real_model_path = model_base_dir 
                else:
                    raise Exception(f"下载失败: {str(dl_err)}")

            self.monitor_signal.emit(False, "", 0)
            if not self.is_running: return
            
            self.stage_signal.emit("加载中 {0}%") 
            self.progress_signal.emit(40)

            # --- 阶段 2: 加载 ---
            self.status_signal.emit("🧠 正在唤醒 AI 引擎...")
            print(f"Init WhisperModel with path: {real_model_path}")
            
            try:
                # 🔥 Ultra 9 防崩配置 🔥
                # 1. float32: 兼容性最好
                # 2. cpu_threads=1: 初始化时单线程，避免 MKL 冲突
                model = WhisperModel(
                    real_model_path, 
                    device="cpu", 
                    compute_type="float32",
                    cpu_threads=1, 
                    local_files_only=True 
                )
                print("Model Loaded Successfully!")
            except Exception as load_err:
                print(f"CRASH DURING LOAD: {load_err}")
                traceback.print_exc()
                raise Exception(f"加载失败 (详见黑框/日志)\n错误: {str(load_err)}")

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

            print("Done.")
            self.progress_signal.emit(100)
            self.status_signal.emit("✅ 转换完成！")
            self.result_signal.emit(full_text)

        except Exception as e:
            print(f"Worker Exception: {e}")
            traceback.print_exc()
            self.monitor_signal.emit(False, "", 0)
            self.error_signal.emit(str(e))

    def get_size(self, folder):
        t = 0
        for r, d, f in os.walk(folder):
            for i in f: t += os.path.getsize(os.path.join(r, i))
        return t / (1024*1024)

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
        self.setWindowTitle("❤️ 调试修复版 (Ultra9)")
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
        
        self.import_area = QPushButton("\n📂 点击上传 / 拖拽视频\n(黑框+日志双重监控)\n")
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
        log_path = os.path.join(BASE_DIR, "crash.log")
        QMessageBox.warning(self, "错误", f"发生错误: {msg}\n\n详细日志已保存至:\n{log_path}")
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