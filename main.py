import sys
import os
import platform
import shutil
import traceback 
import time

# ==============================================================================
# 🔧 调试模式配置
# ==============================================================================

# 1. 允许 Intel 库重复
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 2. 强制国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("***************************************************")
print("          正在启动调试模式 (Debug Mode)           ")
print("***************************************************")
print(f"Python: {sys.version}")
print(f"System: {platform.platform()}")
print(f"Processor: {platform.processor()}")

from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QTextEdit, QProgressBar, QMessageBox, QFileDialog, 
                             QFrame, QGridLayout)
# 🔥 修复点：补充导入 QRect, QRectF
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QRectF
from PyQt6.QtGui import QFont, QColor, QPainter, QPainterPath

# 尝试导入核心库
try:
    print("正在导入 faster_whisper...")
    from faster_whisper import WhisperModel
    print("正在导入 huggingface_hub...")
    from huggingface_hub import snapshot_download
    HAS_WHISPER = True
    print("✅ 核心库导入成功")
except ImportError as e:
    print(f"❌ 核心库缺失: {e}")
    HAS_WHISPER = False
except Exception as e:
    print(f"❌ 导入时发生未知错误: {e}")
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
        
        # 这里的 QRectF 之前漏了 import，现在已修复
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

# === 核心工作线程 ===
class WorkThread(QThread):
    status_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    stage_signal = pyqtSignal(str)
    result_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, video_path, model_code):
        super().__init__()
        self.video_path = video_path
        self.model_code = model_code
        self.repo_id = MODEL_MAP[model_code]
        self.is_running = True

    def run(self):
        print("\n--- 工作线程启动 ---")
        try:
            if getattr(sys, 'frozen', False):
                base_dir = os.path.dirname(sys.executable)
            else:
                base_dir = os.path.dirname(os.path.abspath(__file__))
            
            models_root = os.path.join(base_dir, "models")
            model_dir = os.path.join(models_root, f"models--{self.repo_id.replace('/', '--')}")

            print(f"模型目标路径: {model_dir}")

            # --- 阶段 1: 下载 ---
            self.status_signal.emit(f"⏳ 正在检查模型文件...")
            print("准备调用 snapshot_download...")
            
            try:
                snapshot_download(
                    repo_id=self.repo_id,
                    repo_type="model",
                    local_dir=model_dir,
                    resume_download=True,
                    max_workers=1
                )
                print("✅ 下载/校验完成")
            except Exception as e:
                print("❌ 下载失败")
                traceback.print_exc()
                raise e

            self.stage_signal.emit("加载中 {0}%") 
            self.progress_signal.emit(40)

            # --- 阶段 2: 加载 ---
            self.status_signal.emit("🧠 正在唤醒 AI 引擎...")
            print(f"正在加载模型... 路径: {model_dir}")
            
            try:
                # 显式指定 float32 保证最大兼容性
                model = WhisperModel(
                    model_dir, 
                    device="cpu", 
                    compute_type="float32",  
                    local_files_only=True 
                )
                print("✅ 模型加载进内存成功！")
            except Exception as e:
                print("❌ 模型加载失败！")
                traceback.print_exc()
                raise e

            self.stage_signal.emit("识别中 {0}%")
            self.progress_signal.emit(50)

            # --- 阶段 3: 识别 ---
            self.status_signal.emit("🎧 正在分析...")
            print("开始 transcribe...")
            
            segments, info = model.transcribe(
                self.video_path, beam_size=5, language="zh",
                initial_prompt="这是一段清晰的普通话，请加标点符号。"
            )
            
            print(f"视频时长: {info.duration}秒")
            full_text = ""
            for segment in segments:
                if not self.is_running: return
                full_text += segment.text
                print(f"识别片段: {segment.text}")
                
                if info.duration > 0:
                    progress = 50 + int((segment.end / info.duration) * 48)
                    self.progress_signal.emit(progress)

            print("✅ 任务全部完成")
            self.progress_signal.emit(100)
            self.status_signal.emit("✅ 转换完成！")
            self.result_signal.emit(full_text)

        except Exception as e:
            print("!!! 工作线程捕获到异常 !!!")
            traceback.print_exc()
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
        self.setWindowTitle("❤️ 调试版 - Import 修复")
        self.resize(1100, 700) 
        self.setAcceptDrops(True)
        self.video_path = ""
        self.selected_model = "medium"
        self.worker = None
        self.model_btns = []
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        left_widget = QWidget(); left_layout = QVBoxLayout(left_widget)
        
        self.import_area = QPushButton("\n📂 点击上传 / 拖拽视频\n(调试模式)\n")
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
        f, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "Media (*.mp4 *.mov)")
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
        self.worker.result_signal.connect(self.on_success)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()
    def on_success(self, text):
        self.btn_start.set_progress(100)
        self.text_area.setPlainText(text)
        self.reset_ui()
    def on_error(self, msg):
        self.reset_ui()
        self.lbl_status.setText("❌ 出错")
        QMessageBox.warning(self, "错误", f"发生错误: {msg}\n\n请查看黑框框里的详细报错！")
    def reset_ui(self):
        self.btn_start.stop_processing()
        self.import_area.setEnabled(True)

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! 严重崩溃错误 (FATAL CRASH) !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        traceback.print_exc()
        input("\n按回车键退出程序...")