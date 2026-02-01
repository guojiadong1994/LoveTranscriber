import sys
import os

# ==============================================================================
# 🚀 极简修复：只设置环境变量，依赖文件靠打包脚本搬运
# ==============================================================================
# 1. 允许 OpenMP 重复加载 (防止冲突闪退)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 2. 只有在打包环境下才执行的路径注入 (作为最后一道保险)
if getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    # 将程序根目录加入 DLL 搜索路径 (Python 3.8+ 必须)
    if hasattr(os, 'add_dll_directory'):
        try: os.add_dll_directory(base_dir)
        except: pass
    os.environ['PATH'] = base_dir + os.pathsep + os.environ['PATH']

# ==============================================================================

import shutil
import time
import gc
import requests
import platform
# 延迟导入，防止环境未配置好就崩溃
try:
    from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                                 QLabel, QComboBox, QTextEdit, QProgressBar,
                                 QGroupBox, QMessageBox, QFileDialog, QSplitter)
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt6.QtGui import QFont, QDragEnterEvent, QDropEvent, QGuiApplication, QIcon
    import whisper
    import torch
except ImportError as e:
    # 如果导入失败，弹窗提示 (防止直接闪退看不到报错)
    # 注意：这里只能用 ctypes 弹窗，因为 PyQt 可能还没加载
    import ctypes
    ctypes.windll.user32.MessageBoxW(0, f"启动错误: {str(e)}\n请确保 libiomp5md.dll 在程序根目录！", "错误", 16)
    sys.exit(1)

# === 全局配置 ===
SYSTEM_NAME = platform.system()
IS_MAC = (SYSTEM_NAME == 'Darwin')
UI_FONT_NAME = "PingFang SC" if IS_MAC else "Microsoft YaHei"
FFMPEG_NAME = "ffmpeg" if IS_MAC else "ffmpeg.exe"

MODEL_URLS = {
    "medium (推荐:精准)": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d594138306422b072347d8d909844695d6c5269446f6e469d8/medium.pt",
    "large-v3 (最强:超准)": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c12a951d76f2d12bb234ce3d4160950aed193bbb5427cb9f9d2335/large-v3.pt",
    "base (极速:仅测试)": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small (平衡)": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba4920f77e8deaf87c2546c7d42bca2926851ab63d8dd51895b/small.pt"
}
MODEL_NAMES = {"medium (推荐:精准)": "medium", "large-v3 (最强:超准)": "large-v3", "base (极速:仅测试)": "base", "small (平衡)": "small"}

# === FFmpeg 配置 ===
def setup_ffmpeg_path():
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 优先找同级 bin 目录
    bin_dir = os.path.join(base_dir, "bin")
    ffmpeg_in_bin = os.path.join(bin_dir, FFMPEG_NAME)
    if os.path.exists(ffmpeg_in_bin):
        os.environ["PATH"] += os.pathsep + bin_dir
        return True, "✅ 内置引擎就绪"
    
    # 2. 找系统路径
    if shutil.which("ffmpeg"):
        return True, "✅ 系统引擎就绪"
        
    return False, f"❌ 缺失组件: 请确保 {FFMPEG_NAME} 在 bin 文件夹内"

HAS_FFMPEG, FFMPEG_MSG = setup_ffmpeg_path()

# === 线程与界面逻辑 (保持不变) ===
class ModelLoaderWorker(QThread):
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    def __init__(self, model_key):
        super().__init__()
        self.model_key = model_key
        self.model_name = MODEL_NAMES[model_key]
        self.download_url = MODEL_URLS[model_key]
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        self.download_root = os.path.join(base_dir, "models")
    def run(self):
        if not HAS_FFMPEG:
            self.error_signal.emit(f"无法启动：找不到 {FFMPEG_NAME}")
            return
        if not os.path.exists(self.download_root):
            os.makedirs(self.download_root, exist_ok=True)
        target_file = os.path.join(self.download_root, f"{self.model_name}.pt")
        if not os.path.exists(target_file):
            self.progress_signal.emit(0, f"正在下载 {self.model_name}...")
            try:
                response = requests.get(self.download_url, stream=True, timeout=30)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                with open(target_file, 'wb') as f:
                    last_emit_time = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if time.time() - last_emit_time > 0.1:
                                pct = int((downloaded / total_size) * 100) if total_size > 0 else 0
                                self.progress_signal.emit(pct, f"⬇️ 下载中... {pct}%")
                                last_emit_time = time.time()
                self.progress_signal.emit(100, "校验文件...")
            except Exception as e:
                if os.path.exists(target_file): os.remove(target_file)
                self.error_signal.emit(f"下载失败: {str(e)}")
                return
        try:
            self.progress_signal.emit(100, "🧠 正在载入 AI 引擎...")
            model = whisper.load_model(self.model_name, download_root=self.download_root)
            self.finished_signal.emit(model)
        except Exception as e:
            self.error_signal.emit(f"加载崩溃: {str(e)}")

class TranscribeWorker(QThread):
    finished_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    def __init__(self, model, video_path):
        super().__init__()
        self.model = model
        self.video_path = video_path
        self.is_running = True
    def run(self):
        self.log_signal.emit(f"🎬 读取: {os.path.basename(self.video_path)}")
        self.log_signal.emit("🚀 开始分析语音 (Medium 模型较慢，请耐心)...")
        try:
            result = self.model.transcribe(self.video_path, verbose=False, language='Chinese', initial_prompt="这是一段清晰的普通话视频，请准确识别内容并加上标点符号。")
            if not self.is_running: return 
            self.finished_signal.emit(result['text'].strip())
            self.log_signal.emit("✅ 识别成功！")
        except Exception as e:
            self.error_signal.emit(f"识别出错: {str(e)}")
    def stop(self): self.is_running = False

class TranscriberWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("语音转文字助手 (专属版)")
        self.resize(1000, 700)
        self.setAcceptDrops(True)
        self.model = None
        self.current_video_path = ""
        self.loader_worker = None
        self.trans_worker = None
        self.init_ui()
        self.combo_model.setCurrentIndex(0) 
        self.start_load_model()
    def init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        left_layout = QVBoxLayout()
        title = QLabel("🎥 视频转文字神器")
        title.setFont(QFont(UI_FONT_NAME, 20, QFont.Weight.Bold))
        left_layout.addWidget(title)
        self.lbl_env = QLabel(FFMPEG_MSG)
        self.lbl_env.setStyleSheet("color: green; font-weight: bold;" if HAS_FFMPEG else "color: red; background: #ffe6e6;")
        left_layout.addWidget(self.lbl_env)
        
        grp_model = QGroupBox("⚙️ 引擎设置")
        l_model = QVBoxLayout()
        self.combo_model = QComboBox()
        self.combo_model.addItems(list(MODEL_URLS.keys()))
        self.combo_model.currentIndexChanged.connect(self.on_model_changed)
        l_model.addWidget(self.combo_model)
        self.dl_progress = QProgressBar()
        l_model.addWidget(self.dl_progress)
        grp_model.setLayout(l_model)
        left_layout.addWidget(grp_model)

        self.grp_file = QGroupBox("1. 导入视频")
        l_file = QVBoxLayout()
        self.btn_select = QPushButton("📂 点击选择视频")
        self.btn_select.setFixedHeight(60)
        self.btn_select.clicked.connect(self.select_video)
        self.lbl_path = QLabel("等待导入...")
        l_file.addWidget(self.btn_select)
        l_file.addWidget(self.lbl_path)
        self.grp_file.setLayout(l_file)
        left_layout.addWidget(self.grp_file)

        self.btn_run = QPushButton("✨ 开始识别")
        self.btn_run.setFixedHeight(60)
        self.btn_run.setEnabled(False) 
        self.btn_run.clicked.connect(self.start_transcribe)
        left_layout.addWidget(self.btn_run)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        left_layout.addWidget(self.log_area)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("📝 识别结果:"))
        self.result_area = QTextEdit()
        self.result_area.setFont(QFont(UI_FONT_NAME, 13))
        right_layout.addWidget(self.result_area)
        self.btn_copy = QPushButton("📋 一键复制")
        self.btn_copy.setFixedHeight(50)
        self.btn_copy.clicked.connect(self.copy_result)
        right_layout.addWidget(self.btn_copy)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def on_model_changed(self):
        self.btn_run.setEnabled(False)
        self.start_load_model()
    def start_load_model(self):
        if self.loader_worker: self.loader_worker.terminate()
        self.loader_worker = ModelLoaderWorker(self.combo_model.currentText())
        self.loader_worker.progress_signal.connect(lambda v, m: (self.dl_progress.setValue(v), self.dl_progress.setFormat(m)))
        self.loader_worker.finished_signal.connect(self.on_model_loaded)
        self.loader_worker.error_signal.connect(lambda m: QMessageBox.critical(self, "错误", m))
        self.loader_worker.start()
    def on_model_loaded(self, model):
        self.model = model
        self.dl_progress.setValue(100)
        self.log("模型加载成功！")
        self.check_ready_state()
    def check_ready_state(self):
        if self.model and self.current_video_path and os.path.exists(self.current_video_path):
            self.btn_run.setEnabled(True)
    def dragEnterEvent(self, e): e.accept() if e.mimeData().hasUrls() else e.ignore()
    def dropEvent(self, e): self.set_video(e.mimeData().urls()[0].toLocalFile())
    def select_video(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "Media (*.mp4 *.mov *.avi *.mp3)")
        if f: self.set_video(f)
    def set_video(self, path):
        self.current_video_path = path
        self.lbl_path.setText(f"已就绪: {os.path.basename(path)}")
        self.check_ready_state()
    def start_transcribe(self):
        self.btn_run.setEnabled(False)
        self.result_area.clear()
        self.trans_worker = TranscribeWorker(self.model, self.current_video_path)
        self.trans_worker.log_signal.connect(self.log)
        self.trans_worker.finished_signal.connect(self.on_transcribe_finished)
        self.trans_worker.error_signal.connect(lambda m: QMessageBox.critical(self, "错误", m))
        self.trans_worker.start()
    def on_transcribe_finished(self, text):
        self.result_area.setPlainText(text)
        self.btn_run.setEnabled(True)
        try: QApplication.beep()
        except: pass
        QMessageBox.information(self, "完成", "识别完成！")
    def copy_result(self):
        QGuiApplication.clipboard().setText(self.result_area.toPlainText())
        self.btn_copy.setText("✅ 已复制！")
        QTimer.singleShot(1500, lambda: self.btn_copy.setText("📋 一键复制"))
    def log(self, msg): self.log_area.append(msg)
    def closeEvent(self, e):
        if self.loader_worker: self.loader_worker.terminate()
        if self.trans_worker: self.trans_worker.terminate()
        gc.collect()
        e.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = TranscriberWindow()
    win.show()
    sys.exit(app.exec())