import sys
import os
import platform
import shutil
import traceback
import time
import ctypes

# ==============================================================================
# 🛡️ 0. Intel Ultra 9 专属核心防爆补丁
# ==============================================================================

# 【核心中的核心】禁止 OpenMP 绑定核心
# Ultra 9 是大小核架构，OpenMP 默认的绑定策略会导致内存访问越界(Access Violation)
# 这句代码通常能直接根治 0xC0000005 错误
os.environ["KMP_AFFINITY"] = "disabled"

# 强制降级指令集 (保留，作为双重保险)
os.environ["MKL_ENABLE_INSTRUCTIONS"] = "AVX2"

# 限制线程数 (初始化阶段)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ⚠️ 注意：我移除了 KMP_DUPLICATE_LIB_OK。
# 如果运行报错说 "OMP: Error #15: Initializing libiomp5md.dll..."
# 那就说明是 DLL 重复了，我们需要删文件，而不是改代码。

# ==============================================================================
# 🛡️ 1. 日志与目录配置
# ==============================================================================
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_FILE = os.path.join(BASE_DIR, "crash.log")
MODELS_ROOT = os.path.join(BASE_DIR, "models")

import faulthandler
try:
    log_fs = open(LOG_FILE, "w", encoding="utf-8", buffering=1)
    sys.stdout = log_fs
    sys.stderr = log_fs
    faulthandler.enable(file=log_fs, all_threads=True)
    print(f"===== START {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
    print("Fix: KMP_AFFINITY=disabled")
except: pass

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QTextEdit, QProgressBar, QMessageBox, QFileDialog, 
                             QFrame, QGridLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QRectF
from PyQt6.QtGui import QFont, QColor, QPainter, QPainterPath

try:
    print("Importing faster_whisper...")
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download
    HAS_WHISPER = True
    print("✅ Imported")
except Exception as e:
    print(f"❌ Import failed: {e}")
    HAS_WHISPER = False

# === 全局配置 ===
IS_MAC = (platform.system() == 'Darwin')
UI_FONT = "Microsoft YaHei" if not IS_MAC else "PingFang SC"

MODEL_MAP = { "medium": "systran/faster-whisper-medium", "base": "systran/faster-whisper-base", "large-v3": "systran/faster-whisper-large-v3", "small": "systran/faster-whisper-small" }
MODEL_EXPECTED_SIZE = { "medium": 1500, "base": 145, "large-v3": 3050, "small": 480 }
MODEL_OPTIONS = [
    {"name": "🌟 推荐模式", "desc": "精准与速度平衡", "code": "medium", "color": "#2ecc71"},
    {"name": "🚀 极速模式", "desc": "速度最快", "code": "base", "color": "#3498db"},
    {"name": "🧠 深度模式", "desc": "超准 but 稍慢", "code": "large-v3", "color": "#00cec9"},
    {"name": "⚡ 省电模式", "desc": "轻量级", "code": "small", "color": "#1abc9c"}
]

# === 组件定义 ===
class ProgressButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self._progress = 0.0
        self._is_processing = False
        self.default_text = text
        self.format_str = "运行中 {0}%" 
        self._custom_text = None 
        self.setStyleSheet("QPushButton { background-color: #0078d7; color: white; border-radius: 30px; font-weight: bold; font-size: 20px; } QPushButton:disabled { background-color: #cccccc; color: #888; }")
    def set_progress(self, value):
        self._progress = float(value)
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
        if not self._is_processing: super().paintEvent(event); return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect(); rectf = QRectF(rect)
        painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(QColor("#f0f0f0")); painter.drawRoundedRect(rectf, 30, 30)
        if self._progress > 0:
            prog_width = max(30, (rect.width() * (self._progress / 100.0)))
            path = QPainterPath(); path.addRoundedRect(rectf, 30, 30); painter.setClipPath(path)
            painter.setBrush(QColor("#0078d7")); painter.drawRect(0, 0, int(prog_width), int(rect.height())); painter.setClipping(False)
        painter.setPen(QColor("#333") if self._progress < 55 else QColor("white"))
        font = self.font(); font.setPointSize(16); painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._custom_text if self._custom_text else self.format_str.format(int(self._progress)))

class DownloadMonitor(QThread):
    progress_update = pyqtSignal(int, int, int)
    def __init__(self, target_folder, expected_size_mb):
        super().__init__(); self.target_folder = target_folder; self.expected_size_mb = expected_size_mb; self.is_running = True
    def run(self):
        while self.is_running:
            try:
                total = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fn in os.walk(self.target_folder) for f in fn if not f.endswith(".lock"))
                mb = int(total / 1048576)
                pct = min(39, int((mb / self.expected_size_mb) * 39)) if self.expected_size_mb > 0 else 0
                self.progress_update.emit(mb, self.expected_size_mb, pct)
            except: pass
            time.sleep(0.5)
    def stop(self): self.is_running = False

class WorkThread(QThread):
    status_signal = pyqtSignal(str); progress_signal = pyqtSignal(int); stage_signal = pyqtSignal(str); result_signal = pyqtSignal(str); error_signal = pyqtSignal(str); monitor_signal = pyqtSignal(bool, str, int)
    def __init__(self, video_path, model_code): super().__init__(); self.video_path = video_path; self.model_code = model_code; self.repo_id = MODEL_MAP[model_code]; self.is_running = True
    def run(self):
        try:
            models_root = os.path.join(BASE_DIR, "models"); os.makedirs(models_root, exist_ok=True)
            model_base_dir = os.path.join(models_root, f"models--{self.repo_id.replace('/', '--')}")
            
            self.status_signal.emit(f"⏳ 正在校验/下载模型...")
            expected_mb = MODEL_EXPECTED_SIZE.get(self.model_code, 1000)
            self.monitor_signal.emit(True, model_base_dir, expected_mb)

            try:
                print("Snapshot download...")
                real_model_path = snapshot_download(repo_id=self.repo_id, repo_type="model", local_dir=model_base_dir, resume_download=True, max_workers=1)
                print(f"Path: {real_model_path}")
            except Exception as e:
                print(f"DL Error: {e}"); self.monitor_signal.emit(False, "", 0)
                if os.path.exists(model_base_dir): real_model_path = model_base_dir
                else: raise Exception(f"下载失败: {e}")

            self.monitor_signal.emit(False, "", 0); 
            if not self.is_running: return
            self.stage_signal.emit("加载中 {0}%"); self.progress_signal.emit(40)

            self.status_signal.emit("🧠 正在唤醒 AI 引擎...")
            try:
                # 🔥 改回 int8 (默认)，配合 KMP_AFFINITY=disabled 使用
                model = WhisperModel(real_model_path, device="cpu", compute_type="int8", cpu_threads=4, local_files_only=True)
                print("Model Loaded!")
            except Exception as e:
                print(f"LOAD CRASH: {e}"); traceback.print_exc()
                raise Exception(f"加载崩溃: {e}")

            if not self.is_running: return
            self.stage_signal.emit("识别中 {0}%"); self.progress_signal.emit(50)
            self.status_signal.emit("🎧 正在分析..."); 
            
            segments, info = model.transcribe(self.video_path, beam_size=5, language="zh", initial_prompt="这是一段清晰的普通话，请加标点符号。")
            full_text = ""
            for segment in segments:
                if not self.is_running: return
                full_text += segment.text
                print(f"Seg: {segment.text}")
                if info.duration > 0: self.progress_signal.emit(50 + int((segment.end / info.duration) * 48))
            
            self.progress_signal.emit(100); self.status_signal.emit("✅ 完成！"); self.result_signal.emit(full_text)
        except Exception as e:
            print(f"Err: {e}"); traceback.print_exc(); self.monitor_signal.emit(False, "", 0); self.error_signal.emit(str(e))
    def stop(self): self.is_running = False

class ModelCard(QPushButton):
    def __init__(self, title, desc, code, color, parent=None):
        super().__init__(parent); self.code = code; self.default_color = color; self.setCheckable(True); self.setFixedHeight(100)
        layout = QVBoxLayout(self); l1 = QLabel(title); l1.setFont(QFont(UI_FONT, 15, QFont.Weight.Bold)); layout.addWidget(l1)
        l2 = QLabel(desc); l2.setFont(QFont(UI_FONT, 13)); layout.addWidget(l2)
        self.update_style(False)
    def update_style(self, s): self.setStyleSheet(f"QPushButton {{ background-color: {self.default_color}15; border: 3px solid {self.default_color}; border-radius: 12px; }}" if s else "QPushButton { background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 12px; }")

class MainWindow(QWidget):
    def __init__(self): super().__init__(); self.setWindowTitle("❤️ 专属助手 (Ultra9 最终修复)"); self.resize(1100, 700); self.setAcceptDrops(True); self.video_path = ""; self.selected_model = "medium"; self.worker = None; self.monitor = None; self.model_btns = []; self.init_ui()
    def init_ui(self):
        main = QHBoxLayout(); left = QVBoxLayout(); 
        self.btn_import = QPushButton("\n📂 上传视频\n(黑框日志版)\n"); self.btn_import.setFixedHeight(140); self.btn_import.clicked.connect(self.sel_video); left.addWidget(self.btn_import)
        grid = QGridLayout(); 
        for i, m in enumerate(MODEL_OPTIONS): b = ModelCard(m["name"], m["desc"], m["code"], m["color"]); b.clicked.connect(lambda c, x=b: self.on_clk(x)); grid.addWidget(b, i//2, i%2); self.model_btns.append(b)
        left.addLayout(grid); self.on_clk(self.model_btns[0])
        self.lbl_stat = QLabel("准备就绪"); left.addWidget(self.lbl_stat)
        self.btn_start = ProgressButton("开始转换"); self.btn_start.setFixedHeight(60); self.btn_start.setEnabled(False); self.btn_start.clicked.connect(self.start); left.addWidget(self.btn_start)
        
        right = QVBoxLayout(); self.txt = QTextEdit(); right.addWidget(self.txt)
        btn_cp = QPushButton("📋 复制"); btn_cp.clicked.connect(self.txt.selectAll); btn_cp.clicked.connect(self.txt.copy); right.addWidget(btn_cp)
        
        w_l = QWidget(); w_l.setLayout(left); w_r = QWidget(); w_r.setLayout(right)
        main.addWidget(w_l, 4); main.addWidget(w_r, 6); self.setLayout(main)

    def on_clk(self, b): 
        for x in self.model_btns: x.setChecked(x==b); x.update_style(x==b)
        self.selected_model = b.code
    def dragEnterEvent(self, e): e.accept() if e.mimeData().hasUrls() else e.ignore()
    def dropEvent(self, e): self.load(e.mimeData().urls()[0].toLocalFile())
    def sel_video(self): f, _ = QFileDialog.getOpenFileName(self, "选文件", "", "Media (*.mp4 *.mov *.avi *.mp3)"); self.load(f) if f else None
    def load(self, p): self.video_path = p; self.btn_import.setText(f"已加载: {os.path.basename(p)}"); self.btn_start.setEnabled(True)
    def start(self): 
        self.btn_import.setEnabled(False); self.btn_start.start_processing(); self.worker = WorkThread(self.video_path, self.selected_model)
        self.worker.status_signal.connect(self.lbl_stat.setText); self.worker.progress_signal.connect(self.btn_start.set_progress)
        self.worker.stage_signal.connect(self.btn_start.set_format); self.worker.monitor_signal.connect(self.mon); self.worker.result_signal.connect(self.ok); self.worker.error_signal.connect(self.err); self.worker.start()
    def mon(self, s, p, z): 
        if s: self.monitor = DownloadMonitor(p, z); self.monitor.progress_update.connect(lambda c,t,p: self.btn_start.set_text_override(f"下载 {c}M/{t}M")); self.monitor.start()
        elif self.monitor: self.monitor.stop()
    def ok(self, t): self.btn_start.set_progress(100); self.txt.setPlainText(t); self.btn_import.setEnabled(True); self.btn_start.stop_processing()
    def err(self, m): self.btn_import.setEnabled(True); self.btn_start.stop_processing(); self.lbl_stat.setText("❌ 出错"); QMessageBox.warning(self, "错误", f"错误: {m}\n看日志 crash.log")

if __name__ == "__main__": app = QApplication(sys.argv); w = MainWindow(); w.show(); sys.exit(app.exec())