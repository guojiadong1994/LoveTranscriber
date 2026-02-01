import sys
import os
import platform
import shutil
import traceback
import time
import socket

# ==============================================================================
# 🔧 调试模式配置
# ==============================================================================

# 1. 解决 Intel 库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 2. 强制国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 3. 官方禁言 (防闪退)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

print("***************************************************")
print("          正在启动调试模式 (Debug Mode)           ")
print("***************************************************")
print(f"Python: {sys.version}")
print(f"System: {platform.platform()}")

# 引入所有必要的 UI 库
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QTextEdit, QProgressBar, QMessageBox, QFileDialog, 
                             QFrame, QGridLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QRectF
from PyQt6.QtGui import QFont, QColor, QPainter, QPainterPath

# 尝试导入核心 AI 库
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

MODEL_EXPECTED_SIZE = {
    "medium": 1500,
    "base": 145,
    "large-v3": 3050,
    "small": 480
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
        rectf = QRectF(rect) # 之前这里报错是因为没导入 QRectF，现在已修复

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

# === 监控线程 ===
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
        print(f"--- 任务启动: {self.model_code} ---")
        if not HAS_WHISPER:
            self.error_signal.emit("错误：环境不完整，缺少 faster-whisper")
            return

        try:
            if getattr(sys, 'frozen', False):
                base_dir = os.path.dirname(sys.executable)
            else:
                base_dir = os.path.dirname(os.path.abspath(__file__))
            
            models_root = os.path.join(base_dir, "models")
            model_dir = os.path.join(models_root, f"models--{self.repo_id.replace('/', '--')}")
            print(f"模型路径: {model_dir}")

            # --- 阶段 1: 下载 ---
            self.status_signal.emit(f"⏳ 正在校验/下载模型...")
            expected_mb = MODEL_EXPECTED_SIZE.get(self.model_code, 1000)
            self.monitor_signal.emit(True, model_dir, expected_mb)

            try:
                print("开始调用 snapshot_download...")
                snapshot_download(
                    repo_id=self.repo_id,
                    repo_type="model",
                    local_dir=model_dir,
                    resume_download=True,
                    max_workers=1
                )
                print("下载逻辑执行完毕")
            except Exception as dl_err:
                print(f"下载出错: {dl_err}")
                self.monitor_signal.emit(False, "", 0)
                if os.path.exists(model_dir) and self.get_size(model_dir) > (expected_mb * 0.8):
                    print("本地缓存似乎可用，尝试继续...")
                    self.status_signal.emit("⚠️ 网络微恙，尝试使用本地缓存...")
                else:
                    raise Exception(f"下载失败: {str(dl_err)}")

            self.monitor_signal.emit(False, "", 0)
            if not self.is_running: return
            
            self.stage_signal.emit("加载中 {0}%") 
            self.progress_signal.emit(40)

            # --- 阶段 2: 加载 ---
            self.status_signal.emit("🧠 正在唤醒 AI 引擎 (请勿关闭)...")
            print("开始加载模型 WhisperModel...")
            
            try:
                # 兼容模式：float32
                model = WhisperModel(
                    model_dir, 
                    device="cpu", 
                    compute_type="float32",
                    local_files_only=True 
                )
                print("模型加载成功！")
            except Exception as load_err:
                print(f"模型加载崩溃: {load_err}")
                traceback.print_exc()
                if os.path.exists(model_dir):
                    try: shutil.rmtree(model_dir)
                    except: pass
                raise Exception(f"模型加载失败(已重置)，请重试。\n错误: {str(load_err)}")

            if not self.is_running: return
            self.stage_signal.emit("识别中 {0}%")
            self.progress_signal.emit(50)

            # --- 阶段 3: 识别 ---
            self.status_signal.emit("🎧 正在分析...")
            print("开始转录...")
            
            segments, info = model.transcribe(
                self.video_path, beam_size=5, language="zh",
                initial_prompt="这是一段清晰的普通话，请加标点符号。"
            )
            
            full_text = ""
            total_duration = info.duration
            print(f"视频时长: {total_duration}")

            for segment in segments:
                if not self.is_running: return
                full_text += segment.text
                print(f"识别片段: {segment.text}")
                
                if total_duration > 0:
                    progress = 50 + int((segment.end / total_duration) * 48)
                    self.progress_signal.emit(progress)

            print("转录完成")
            self.progress_signal.emit(100)
            self.status_signal.emit("✅ 转换完成！")
            self.result_signal.emit(full_text)

        except Exception as e:
            print(f"未捕获异常: {e}")
            traceback.print_exc()
            self.monitor_signal.emit(False, "", 0)
            self.error_signal.emit(str(e))

    def get_size(self, folder):
        t = 0
        for r, d, f in os.walk(folder):
            for i in f: t += os.path.getsize(os.path.join(r, i))
        return t / (1024*1024)

    def stop(self): self.is_running = False

# ==============================================================================
# 📦 模型卡片组件 (ModelCard) - 之前你丢失的就是这个类！
# ==============================================================================
class ModelCard(QPushButton):
    def __init__(self, title, desc, code, color, parent=None):
        super().__init__(parent)
        self.code = code
        self.default_color = color
        self.setCheckable(True)
        self.setFixedHeight(100) 
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 12)
        layout.setSpacing(8)
        
        self.lbl_title = QLabel(title)
        self.lbl_title.setFont(QFont(UI_FONT, 15, QFont.Weight.Bold))
        
        self.lbl_desc = QLabel(desc)
        self.lbl_desc.setFont(QFont(UI_FONT, 13))
        self.lbl_desc.setStyleSheet("color: #666;")
        
        layout.addWidget(self.lbl_title)
        layout.addWidget(self.lbl_desc)
        
        self.update_style(False)

    def update_style(self, selected):
        if selected:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {self.default_color}15;
                    border: 3px solid {self.default_color};
                    border-radius: 12px;
                    text-align: left;
                }}
            """)
            self.lbl_title.setStyleSheet(f"color: {self.default_color};")
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #f9f9f9;
                    border: 1px solid #ddd;
                    border-radius: 12px;
                    text-align: left;
                }
                QPushButton:hover { background-color: white; border-color: #bbb; }
            """)
            self.lbl_title.setStyleSheet("color: #333;")

# ==============================================================================
# 🖥️ 主窗口 (MainWindow)
# ==============================================================================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("❤️ 专属语音转文字助手 (调试修复版)")
        self.resize(1100, 700) 
        self.setAcceptDrops(True)
        self.video_path = ""
        self.selected_model = "medium"
        self.worker = None
        self.monitor = None
        self.model_btns = []
        
        self.fake_progress_timer = QTimer()
        self.fake_progress_timer.timeout.connect(self.update_fake_progress)

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(40)

        # 左侧布局
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(25) 

        lbl_step1 = QLabel("第一步：上传视频")
        lbl_step1.setFont(QFont(UI_FONT, 18, QFont.Weight.Bold))
        left_layout.addWidget(lbl_step1)

        self.import_area = QPushButton("\n📂 点击上传 / 拖拽视频\n(再次点击可替换)\n")
        self.import_area.setCursor(Qt.CursorShape.PointingHandCursor)
        self.import_area.setFixedHeight(140) 
        self.import_area.setFont(QFont(UI_FONT, 15))
        self.import_area.setStyleSheet("""
            QPushButton { background-color: #f0f7ff; border: 3px dashed #0078d7; border-radius: 20px; color: #0078d7; }
            QPushButton:hover { background-color: #e0efff; }
        """)
        self.import_area.clicked.connect(self.select_video)
        left_layout.addWidget(self.import_area)

        lbl_step2 = QLabel("第二步：选择识别模型")
        lbl_step2.setFont(QFont(UI_FONT, 18, QFont.Weight.Bold))
        left_layout.addWidget(lbl_step2)

        model_layout = QGridLayout()
        model_layout.setSpacing(15)
        for i, m in enumerate(MODEL_OPTIONS):
            # 🔥 这里之前报错是因为 ModelCard 没有定义，现在已经修复
            btn = ModelCard(m["name"], m["desc"], m["code"], m["color"])
            btn.clicked.connect(lambda checked, b=btn: self.on_model_click(b))
            model_layout.addWidget(btn, i // 2, i % 2)
            self.model_btns.append(btn)
        left_layout.addLayout(model_layout)
        self.on_model_click(self.model_btns[0])

        left_layout.addStretch()

        self.lbl_status = QLabel("准备就绪")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setFont(QFont(UI_FONT, 14))
        self.lbl_status.setStyleSheet("color: #666; font-weight: bold;")
        left_layout.addWidget(self.lbl_status)

        self.btn_start = ProgressButton("开始转换")
        self.btn_start.setFixedHeight(60)
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start.setEnabled(False) 
        self.btn_start.clicked.connect(self.start_process)
        left_layout.addWidget(self.btn_start)

        # 右侧布局
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(15)

        lbl_res = QLabel("📝 转换结果 (可编辑)")
        lbl_res.setFont(QFont(UI_FONT, 16, QFont.Weight.Bold))
        right_layout.addWidget(lbl_res)

        self.text_area = QTextEdit()
        self.text_area.setPlaceholderText("识别的文字会显示在这里...")
        self.text_area.setFont(QFont(UI_FONT, 20)) 
        self.text_area.setStyleSheet("""
            QTextEdit { border: 1px solid #ddd; border-radius: 15px; padding: 20px; background: #fafafa; selection-background-color: #0078d7; line-height: 160%; }
            QTextEdit:focus { background: white; border-color: #0078d7; }
        """)
        right_layout.addWidget(self.text_area)

        self.btn_copy = QPushButton("📋 一键复制全部")
        self.btn_copy.setFixedHeight(60)
        self.btn_copy.setFont(QFont(UI_FONT, 16, QFont.Weight.Bold))
        self.btn_copy.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_copy.setStyleSheet("""
            QPushButton { background: white; color: #333; border: 1px solid #ddd; border-radius: 12px; }
            QPushButton:hover { background: #f5f5f5; border-color: #aaa; }
        """)
        self.btn_copy.clicked.connect(self.copy_text)
        right_layout.addWidget(self.btn_copy)

        main_layout.addWidget(left_widget, 4)
        main_layout.addWidget(right_widget, 6)
        self.setLayout(main_layout)

    def on_model_click(self, clicked_btn):
        for btn in self.model_btns:
            is_target = (btn == clicked_btn)
            btn.setChecked(is_target)
            btn.update_style(is_target)
        self.selected_model = clicked_btn.code

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.accept()
        else: e.ignore()

    def dropEvent(self, e):
        self.load_video(e.mimeData().urls()[0].toLocalFile())

    def select_video(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "Media (*.mp4 *.mov *.avi *.mp3 *.m4a *.wav)")
        if f: self.load_video(f)

    def load_video(self, path):
        self.video_path = path
        name = os.path.basename(path)
        self.import_area.setText(f"\n📄 已就绪：{name}\n(点击可替换)\n")
        self.import_area.setStyleSheet("""
            QPushButton { background-color: #f0fff4; border: 2px solid #2ecc71; border-radius: 20px; color: #27ae60; font-weight: bold; }
            QPushButton:hover { background-color: #dcfce7; }
        """)
        self.lbl_status.setText("视频已加载，请点击开始")
        self.btn_start.setEnabled(True)

    def start_process(self):
        if not self.video_path: return
        self.import_area.setEnabled(False)
        for btn in self.model_btns: btn.setEnabled(False)
        self.text_area.clear()

        self.btn_start.start_processing()
        
        self.worker = WorkThread(self.video_path, self.selected_model)
        self.worker.status_signal.connect(self.lbl_status.setText) 
        self.worker.progress_signal.connect(self.update_progress_val) 
        self.worker.stage_signal.connect(self.update_progress_format) 
        self.worker.monitor_signal.connect(self.handle_monitor_request)
        self.worker.result_signal.connect(self.on_success)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def handle_monitor_request(self, should_start, folder_path, expected_mb):
        if should_start:
            self.fake_progress_timer.stop() 
            if self.monitor: self.monitor.stop()
            self.monitor = DownloadMonitor(folder_path, expected_mb)
            self.monitor.progress_update.connect(self.on_monitor_update)
            self.monitor.start()
        else:
            if self.monitor: self.monitor.stop()
            self.fake_progress_timer.start(100)

    def on_monitor_update(self, current_mb, total_mb, pct):
        msg = f"下载中 {current_mb}MB / {total_mb}MB"
        self.btn_start.set_progress(pct) 
        self.btn_start.set_text_override(msg) 

    def update_progress_val(self, val):
        self.btn_start.set_progress(val)

    def update_progress_format(self, fmt):
        self.btn_start.set_format(fmt) 

    def update_fake_progress(self):
        self.btn_start.auto_creep_progress()

    def on_success(self, text):
        if self.monitor: self.monitor.stop()
        self.fake_progress_timer.stop()
        self.btn_start.set_progress(100)
        self.btn_start.set_text_override("转换完成")
        self.text_area.setPlainText(text)
        self.reset_ui()
        QMessageBox.information(self, "成功", "转换完成！")

    def on_error(self, msg):
        if self.monitor: self.monitor.stop()
        self.fake_progress_timer.stop()
        self.reset_ui()
        self.lbl_status.setText("❌ 发生错误")
        QMessageBox.warning(self, "错误", f"程序遇到了问题:\n{msg}")

    def reset_ui(self):
        self.btn_start.stop_processing()
        self.import_area.setEnabled(True)
        for btn in self.model_btns: btn.setEnabled(True)
        self.lbl_status.setText("准备就绪")

    def copy_text(self):
        txt = self.text_area.toPlainText()
        if txt:
            QApplication.clipboard().setText(txt)
            self.btn_copy.setText("✅ 已复制")
            QTimer.singleShot(1500, lambda: self.btn_copy.setText("📋 一键复制全部"))
    
    def closeEvent(self, event):
        if self.monitor: self.monitor.stop()
        if self.fake_progress_timer.isActive(): self.fake_progress_timer.stop()
        os._exit(0)

# 🔥🔥🔥 程序入口 🔥🔥🔥
if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print("\n\n!!! CRITICAL ERROR !!!")
        traceback.print_exc()
        input("Press Enter to exit...")