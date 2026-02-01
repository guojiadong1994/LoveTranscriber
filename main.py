import sys
import os

# ==================================================
# 🚑 关键修复：DLL 路径强力注入 (放在所有 import 之前)
# ==================================================
if getattr(sys, 'frozen', False):
    # 1. 确定程序所在的根目录
    application_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    
    # 2. 定义可能存放 torch DLL 的所有角落
    # 兼容: 文件夹模式(libs目录)、单文件模式(_MEIPASS)、普通模式
    potential_paths = [
        application_path,
        os.path.join(application_path, 'libs'),                # 你的打包配置用了这个
        os.path.join(application_path, 'libs', 'torch', 'lib'), # PyTorch 的老巢
        os.path.join(application_path, 'torch', 'lib'),
    ]
    
    # 如果是单文件模式，还有个临时目录
    if hasattr(sys, '_MEIPASS'):
        potential_paths.append(sys._MEIPASS)
        potential_paths.append(os.path.join(sys._MEIPASS, 'torch', 'lib'))

    # 3. 暴力注入 PATH 环境变量
    # 把这些路径全部加到系统查找路径的最前面
    new_path = os.environ['PATH']
    for p in potential_paths:
        if p and os.path.exists(p):
            new_path = p + os.pathsep + new_path
    
    os.environ['PATH'] = new_path

# ==================================================

import shutil
import time
import gc
import requests
import platform
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QComboBox, QTextEdit, QProgressBar,
                             QGroupBox, QMessageBox, QFileDialog, QSplitter)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QDragEnterEvent, QDropEvent, QGuiApplication, QIcon

# === 全局配置 ===
SYSTEM_NAME = platform.system()
IS_MAC = (SYSTEM_NAME == 'Darwin')

# 字体适配
UI_FONT_NAME = "PingFang SC" if IS_MAC else "Microsoft YaHei"
# FFmpeg 文件名适配
FFMPEG_NAME = "ffmpeg" if IS_MAC else "ffmpeg.exe"

# 模型配置
MODEL_URLS = {
    "medium (推荐:精准)": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d594138306422b072347d8d909844695d6c5269446f6e469d8/medium.pt",
    "large-v3 (最强:超准)": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c12a951d76f2d12bb234ce3d4160950aed193bbb5427cb9f9d2335/large-v3.pt",
    "base (极速:仅测试)": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small (平衡)": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba4920f77e8deaf87c2546c7d42bca2926851ab63d8dd51895b/small.pt"
}

MODEL_NAMES = {
    "medium (推荐:精准)": "medium",
    "large-v3 (最强:超准)": "large-v3",
    "base (极速:仅测试)": "base",
    "small (平衡)": "small"
}

# === 0. 自动配置 FFmpeg ===
def setup_ffmpeg_path():
    """检测 bin 目录下的 ffmpeg 并注入环境变量"""
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 检查 bin 目录
    bin_dir = os.path.join(base_dir, "bin")
    ffmpeg_in_bin = os.path.join(bin_dir, FFMPEG_NAME)
    
    if os.path.exists(ffmpeg_in_bin):
        os.environ["PATH"] += os.pathsep + bin_dir
        return True, "✅ 内置引擎就绪 (bin)"

    # 2. 检查根目录
    ffmpeg_in_root = os.path.join(base_dir, FFMPEG_NAME)
    if os.path.exists(ffmpeg_in_root):
        os.environ["PATH"] += os.pathsep + base_dir
        return True, "✅ 根目录引擎就绪"
    
    # 3. 检查系统
    if shutil.which("ffmpeg"):
        return True, "✅ 系统引擎就绪"
        
    return False, f"❌ 缺失组件: 请确保 {FFMPEG_NAME} 在 bin 文件夹内"

HAS_FFMPEG, FFMPEG_MSG = setup_ffmpeg_path()

# 延迟导入 whisper，防止启动时卡死
try:
    import whisper
    import torch
except ImportError:
    whisper = None


# === 1. 下载/加载模型线程 ===
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

        if not whisper:
            self.error_signal.emit("环境错误：未安装 openai-whisper")
            return

        if not os.path.exists(self.download_root):
            os.makedirs(self.download_root, exist_ok=True)
        
        target_file = os.path.join(self.download_root, f"{self.model_name}.pt")

        # --- 下载逻辑 ---
        if not os.path.exists(target_file):
            self.progress_signal.emit(0, f"正在下载 {self.model_name} 模型 (首次运行)...")
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
                self.error_signal.emit(f"下载失败: {str(e)}\n建议手动下载模型文件放入 models 文件夹")
                return
        else:
            self.progress_signal.emit(100, "检测到本地模型，准备加载...")

        # --- 加载逻辑 ---
        try:
            self.progress_signal.emit(100, "🧠 正在载入 AI 引擎 (请稍候)...")
            # 这里的 download_root 很重要，指定模型寻找路径
            model = whisper.load_model(self.model_name, download_root=self.download_root)
            self.finished_signal.emit(model)
        except Exception as e:
            self.error_signal.emit(f"加载崩溃: {str(e)}")


# === 2. 识别线程 ===
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
        self.log_signal.emit("🚀 开始分析语音 (Medium 模型较慢但精准，请耐心)...")
        
        try:
            # 关键参数：fp16=False 兼容 Mac CPU
            result = self.model.transcribe(
                self.video_path, 
                verbose=False, 
                language='Chinese',
                initial_prompt="这是一段清晰的普通话视频，请准确识别内容并加上标点符号。"
            )
            
            if not self.is_running: return 
            
            text = result['text'].strip()
            self.finished_signal.emit(text)
            self.log_signal.emit("✅ 识别成功！")

        except Exception as e:
            self.error_signal.emit(f"识别出错: {str(e)}")

    def stop(self):
        self.is_running = False


# === 3. 主窗口 ===
class TranscriberWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("语音转文字助手 (女朋友专属版 ❤️)")
        self.resize(1000, 700)
        self.setAcceptDrops(True)

        self.model = None
        self.current_video_path = ""
        self.loader_worker = None
        self.trans_worker = None

        self.init_ui()
        
        # 默认加载 Medium
        self.combo_model.setCurrentIndex(0) 
        self.start_load_model()

    def init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # === 左侧面板 ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        
        title = QLabel("🎥 视频转文字神器")
        title.setFont(QFont(UI_FONT_NAME, 20, QFont.Weight.Bold))
        left_layout.addWidget(title)

        # 环境提示
        self.lbl_env = QLabel(FFMPEG_MSG)
        style = "color: green; font-weight: bold;" if HAS_FFMPEG else "color: red; background: #ffe6e6; padding: 5px;"
        self.lbl_env.setStyleSheet(style)
        left_layout.addWidget(self.lbl_env)

        # 模型设置
        grp_model = QGroupBox("⚙️ 引擎设置")
        grp_model.setFont(QFont(UI_FONT_NAME, 10))
        l_model = QVBoxLayout()
        self.combo_model = QComboBox()
        self.combo_model.addItems(list(MODEL_URLS.keys()))
        self.combo_model.currentIndexChanged.connect(self.on_model_changed)
        l_model.addWidget(QLabel("识别模型:"))
        l_model.addWidget(self.combo_model)
        
        self.dl_progress = QProgressBar()
        self.dl_progress.setValue(0)
        self.dl_progress.setTextVisible(True)
        self.dl_progress.setStyleSheet("QProgressBar { height: 6px; border-radius: 3px; } QProgressBar::chunk { background-color: #0078d7; }")
        l_model.addWidget(self.dl_progress)
        grp_model.setLayout(l_model)
        left_layout.addWidget(grp_model)

        # 视频导入区
        self.grp_file = QGroupBox("1. 导入视频")
        self.grp_file.setFont(QFont(UI_FONT_NAME, 10))
        l_file = QVBoxLayout()
        self.btn_select = QPushButton("📂 拖拽视频到这里\n或点击选择文件")
        self.btn_select.setFixedHeight(80)
        self.btn_select.setFont(QFont(UI_FONT_NAME, 11))
        self.btn_select.setStyleSheet("background-color: #f5f5f5; border: 2px dashed #aaa; border-radius: 10px;")
        self.btn_select.clicked.connect(self.select_video)
        
        self.lbl_path = QLabel("等待导入...")
        self.lbl_path.setWordWrap(True)
        self.lbl_path.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_path.setStyleSheet("color: #666;")
        
        l_file.addWidget(self.btn_select)
        l_file.addWidget(self.lbl_path)
        self.grp_file.setLayout(l_file)
        left_layout.addWidget(self.grp_file)

        # 开始按钮
        self.btn_run = QPushButton("✨ 开始识别")
        self.btn_run.setFont(QFont(UI_FONT_NAME, 14, QFont.Weight.Bold))
        self.btn_run.setFixedHeight(60)
        self.btn_run.setStyleSheet("""
            QPushButton { background-color: #ccc; color: white; border-radius: 8px; }
            QPushButton:enabled { background-color: #0078d7; }
            QPushButton:enabled:hover { background-color: #0063b1; }
        """)
        self.btn_run.setEnabled(False) 
        self.btn_run.clicked.connect(self.start_transcribe)
        left_layout.addWidget(self.btn_run)

        # 日志区
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("background: #2b2b2b; color: #eee; border-radius: 5px; font-size: 11px;")
        left_layout.addWidget(self.log_area)

        # === 右侧面板 ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)

        right_layout.addWidget(QLabel("📝 识别结果 (可直接编辑):"))
        
        self.result_area = QTextEdit()
        self.result_area.setFont(QFont(UI_FONT_NAME, 13))
        self.result_area.setStyleSheet("padding: 10px; line-height: 1.6; border: 1px solid #ddd; border-radius: 5px;")
        right_layout.addWidget(self.result_area)

        self.btn_copy = QPushButton("📋 确认无误，一键复制")
        self.btn_copy.setFixedHeight(50)
        self.btn_copy.setFont(QFont(UI_FONT_NAME, 12, QFont.Weight.Bold))
        self.btn_copy.setStyleSheet("""
            QPushButton { background-color: #28a745; color: white; border-radius: 8px; }
            QPushButton:hover { background-color: #218838; }
        """)
        self.btn_copy.clicked.connect(self.copy_result)
        right_layout.addWidget(self.btn_copy)

        # 分割布局
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    # --- 逻辑功能 ---

    def on_model_changed(self):
        self.btn_run.setEnabled(False)
        self.start_load_model()

    def start_load_model(self):
        model_key = self.combo_model.currentText()
        if self.loader_worker: self.loader_worker.terminate()
        
        self.log(f"--- 准备加载模型: {model_key} ---")
        self.loader_worker = ModelLoaderWorker(model_key)
        self.loader_worker.progress_signal.connect(self.update_dl_progress)
        self.loader_worker.error_signal.connect(self.on_load_error)
        self.loader_worker.finished_signal.connect(self.on_model_loaded)
        self.loader_worker.start()

    def update_dl_progress(self, val, msg):
        self.dl_progress.setValue(val)
        self.dl_progress.setFormat(msg)
        if val == 100 and "加载" in msg: self.log(msg)

    def on_model_loaded(self, model):
        self.model = model
        self.dl_progress.setFormat("✅ 就绪")
        self.dl_progress.setValue(100)
        self.log("模型加载成功！")
        self.check_ready_state()

    def on_load_error(self, msg):
        self.dl_progress.setFormat("❌ 失败")
        QMessageBox.critical(self, "错误", msg)
        self.log(msg)

    def check_ready_state(self):
        if self.model and self.current_video_path and os.path.exists(self.current_video_path):
            self.btn_run.setEnabled(True)
            self.btn_run.setText("✨ 开始识别")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls(): event.accept()
        else: event.ignore()

    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files: self.set_video(files[0])

    def select_video(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "Media (*.mp4 *.mov *.avi *.mp3 *.wav *.m4a)")
        if f: self.set_video(f)

    def set_video(self, path):
        self.current_video_path = path
        self.lbl_path.setText(f"已就绪: {os.path.basename(path)}")
        self.lbl_path.setStyleSheet("color: #0078d7; font-weight: bold;")
        self.log(f"已选中: {path}")
        self.check_ready_state()

    def start_transcribe(self):
        self.btn_run.setEnabled(False)
        self.btn_run.setText("⏳ 识别中...")
        self.result_area.clear()
        
        self.trans_worker = TranscribeWorker(self.model, self.current_video_path)
        self.trans_worker.log_signal.connect(self.log)
        self.trans_worker.error_signal.connect(lambda m: QMessageBox.critical(self, "错误", m))
        self.trans_worker.finished_signal.connect(self.on_transcribe_finished)
        self.trans_worker.start()

    def on_transcribe_finished(self, text):
        self.result_area.setPlainText(text)
        self.btn_run.setEnabled(True)
        self.btn_run.setText("✨ 再次识别")
        
        try:
            QApplication.beep()
        except:
            pass
        
        QMessageBox.information(self, "完成", "识别完成！请校对后复制。")

    def copy_result(self):
        text = self.result_area.toPlainText()
        if not text: return
        QGuiApplication.clipboard().setText(text)
        self.btn_copy.setText("✅ 已复制！")
        QTimer.singleShot(1500, lambda: self.btn_copy.setText("📋 确认无误，一键复制"))

    def log(self, msg):
        self.log_area.append(msg)
        sb = self.log_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    # 退出时彻底杀进程，防止残留
    def closeEvent(self, event):
        self.log("清理资源...")
        if self.loader_worker: self.loader_worker.terminate()
        if self.trans_worker: self.trans_worker.terminate()
        if self.model:
            del self.model
            self.model = None
        gc.collect()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = TranscriberWindow()
    win.show()
    sys.exit(app.exec())