import sys
import os
import platform
import time

# ==============================================================================
# 🛡️ 防闪退核心补丁 (必须放在最前面)
# ==============================================================================

# 1. 魔法指令：强制使用国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 禁用 TQDM 进度条 (防止底层库打印进度导致无控制台模式崩溃)
os.environ["TQDM_DISABLE"] = "1"

# 3. "黑洞"类：吃掉所有打印信息
class NullWriter:
    def write(self, text): pass
    def flush(self): pass

# 4. 如果是打包环境，强制接管 stdout/stderr
# 这样任何库想打印东西，都会被扔进黑洞，不会因为找不到控制台而闪退
if getattr(sys, 'frozen', False):
    sys.stdout = NullWriter()
    sys.stderr = NullWriter()

# ==============================================================================

from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QTextEdit, QProgressBar, QMessageBox, QFileDialog, 
                             QFrame, QGridLayout, QStyleOptionButton, QStyle)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QRectF
from PyQt6.QtGui import QFont, QColor, QPalette, QPainter, QBrush, QPen, QPainterPath

# === 全局配置 ===
IS_MAC = (platform.system() == 'Darwin')
UI_FONT = "Microsoft YaHei" if not IS_MAC else "PingFang SC"

MODEL_SPECS = {
    "medium":   {"name": "🌟 推荐模式", "desc": "精准与速度平衡", "code": "medium", "color": "#2ecc71", "size": 1500},
    "base":     {"name": "🚀 极速模式", "desc": "速度最快", "code": "base", "color": "#3498db", "size": 150},
    "large-v3": {"name": "🧠 深度模式", "desc": "超准但稍慢", "code": "large-v3", "color": "#00cec9", "size": 3100},
    "small":    {"name": "⚡ 省电模式", "desc": "轻量级", "code": "small", "color": "#1abc9c", "size": 500}
}
MODEL_OPTIONS = list(MODEL_SPECS.values())

# === 自定义：带进度条的按钮 ===
class ProgressButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self._progress = 0.0
        self._is_processing = False
        self.default_text = text
        self.processing_text = "转换中 {0}%"
        self._extra_text = "" 
        
        self.setStyleSheet("""
            QPushButton {
                background-color: #0078d7; 
                color: white; 
                border-radius: 30px;
                font-weight: bold;
                font-size: 20px; 
            }
            QPushButton:hover { background-color: #0063b1; }
            QPushButton:pressed { background-color: #005a9e; }
            QPushButton:disabled { background-color: #cccccc; color: #888; }
        """)

    def set_progress(self, value, text_override=None):
        if value > self._progress:
            self._progress = float(value)
        
        if text_override:
            self.setText(text_override)
        else:
            self.setText(self.processing_text.format(int(self._progress)))
        self.update() 

    def increment_fake_progress(self, amount=0.2):
        if self._progress < 99.0:
            self._progress += amount
            if self._progress > 99.0: self._progress = 99.0
            self.setText(self.processing_text.format(int(self._progress)))
            self.update()

    def start_processing(self):
        self._is_processing = True
        self._progress = 0.0
        self.setEnabled(False) 
        self.update()

    def stop_processing(self):
        self._is_processing = False
        self._progress = 0.0
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

        # 1. 背景
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#f0f0f0"))
        painter.drawRoundedRect(rectf, 30, 30)

        # 2. 进度条
        if self._progress > 0:
            prog_width = (rect.width() * (self._progress / 100.0))
            if prog_width < 30: prog_width = 30
            path = QPainterPath()
            path.addRoundedRect(rectf, 30, 30)
            painter.setClipPath(path)
            painter.setBrush(QColor("#0078d7"))
            painter.drawRect(0, 0, int(prog_width), int(rect.height()))
            painter.setClipping(False)

        # 3. 文字
        painter.setPen(QColor("#333") if self._progress < 55 else QColor("white"))
        font = self.font()
        font.setPointSize(16) 
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.text())


# === 下载监视线程 (加固版) ===
class DownloadMonitor(QThread):
    size_signal = pyqtSignal(int, int)

    def __init__(self, target_folder, estimated_size_mb):
        super().__init__()
        self.target_folder = target_folder
        self.estimated_size_mb = estimated_size_mb
        self.is_running = True

    def get_folder_size(self):
        total_size = 0
        try:
            # 增加 try-except 防止在文件写入瞬间读取导致权限错误
            for dirpath, dirnames, filenames in os.walk(self.target_folder):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    # 忽略快捷方式和临时锁文件
                    if not os.path.islink(fp):
                        try:
                            total_size += os.path.getsize(fp)
                        except OSError:
                            continue 
        except Exception:
            pass
        return total_size / (1024 * 1024) 

    def run(self):
        while self.is_running:
            current_mb = int(self.get_folder_size())
            self.size_signal.emit(current_mb, self.estimated_size_mb)
            time.sleep(0.5) 

    def stop(self):
        self.is_running = False


# === 核心工作线程 ===
class WorkThread(QThread):
    status_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, str)
    result_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, video_path, model_code):
        super().__init__()
        self.video_path = video_path
        self.model_code = model_code
        self.is_running = True
        self.monitor = None

    def run(self):
        # 延迟导包
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            self.error_signal.emit("错误：未检测到 faster-whisper 库！")
            return

        try:
            # 1. 准备路径
            if getattr(sys, 'frozen', False):
                base_dir = os.path.dirname(sys.executable)
            else:
                base_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(base_dir, "models")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            # 2. 启动下载监视器
            target_mb = MODEL_SPECS[self.model_code]["size"]
            self.monitor = DownloadMonitor(model_dir, target_mb)
            self.monitor.size_signal.connect(self.on_download_update)
            self.monitor.start()

            self.status_signal.emit(f"⏳ 正在下载/加载模型 (约 {target_mb} MB)...")
            
            # 3. 开始加载
            # 注意：这里的 download_root 会触发下载
            model = WhisperModel(
                self.model_code, 
                device="cpu", 
                compute_type="int8", 
                download_root=model_dir
            )
            
            # 停止监视
            self.monitor.stop()
            self.monitor.wait()
            
            if not self.is_running: return
            self.progress_signal.emit(20, None) 

            # 4. 分析
            self.status_signal.emit("🎧 正在分析语音内容...")
            
            segments, info = model.transcribe(
                self.video_path, beam_size=5, language="zh",
                initial_prompt="这是一段清晰的普通话，请加标点符号。"
            )

            full_text = ""
            total_duration = info.duration
            current_time = 0

            self.status_signal.emit("📝 正在生成文字...")

            for segment in segments:
                if not self.is_running: return
                full_text += segment.text
                current_time = segment.end
                
                if total_duration > 0:
                    progress = 20 + int((current_time / total_duration) * 78)
                    self.progress_signal.emit(progress, None)

            self.progress_signal.emit(100, None)
            self.status_signal.emit("✅ 转换完成！")
            self.result_signal.emit(full_text)

        except Exception as e:
            if self.monitor: self.monitor.stop()
            self.error_signal.emit(f"出错: {str(e)}")

    def on_download_update(self, current_mb, target_mb):
        if target_mb > 0:
            dl_progress = int((current_mb / target_mb) * 19) 
            if dl_progress > 19: dl_progress = 19
            msg = f"下载中 {current_mb}MB / {target_mb}MB"
            self.progress_signal.emit(dl_progress, msg)

    def stop(self):
        self.is_running = False
        if self.monitor: self.monitor.stop()


# === 模型卡片 ===
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

# === 主窗口 ===
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("❤️ 专属语音转文字助手")
        self.resize(1100, 700) 
        self.setAcceptDrops(True)
        
        self.video_path = ""
        self.selected_model = "medium"
        self.worker = None
        self.model_btns = []
        
        self.fake_progress_timer = QTimer()
        self.fake_progress_timer.timeout.connect(self.update_fake_progress)

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(40)

        # =========== 左侧栏 ===========
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(25) 

        lbl_step1 = QLabel("第一步：上传视频")
        lbl_step1.setFont(QFont(UI_FONT, 18, QFont.Weight.Bold))
        left_layout.addWidget(lbl_step1)

        self.import_area = QPushButton("\n📂 点击上传 / 拖拽视频\n(再次点击可替换)\n")
        self.import_area.setCursor(Qt.CursorShape.PointingHandCursor)
        self.import_area.setFixedHeight(140) 
        self.import_area.setFont(QFont(UI_FONT, 15))
        self.import_area.setStyleSheet("""
            QPushButton {
                background-color: #f0f7ff;
                border: 3px dashed #0078d7;
                border-radius: 20px;
                color: #0078d7;
            }
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

        # =========== 右侧栏 ===========
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(15)

        lbl_res = QLabel("📝 转换结果 (可编辑)")
        lbl_res.setFont(QFont(UI_FONT, 16, QFont.Weight.Bold))
        right_layout.addWidget(lbl_res)

        self.text_area = QTextEdit()
        self.text_area.setPlaceholderText("识别的文字会显示在这里...")
        self.text_area.setFont(QFont(UI_FONT, 20)) 
        self.text_area.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 15px;
                padding: 20px;
                background-color: #fafafa;
                selection-background-color: #0078d7;
                line-height: 160%;
            }
            QTextEdit:focus {
                background-color: white;
                border-color: #0078d7;
            }
        """)
        right_layout.addWidget(self.text_area)

        self.btn_copy = QPushButton("📋 一键复制全部")
        self.btn_copy.setFixedHeight(60)
        self.btn_copy.setFont(QFont(UI_FONT, 16, QFont.Weight.Bold))
        self.btn_copy.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_copy.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: #333;
                border: 1px solid #ddd;
                border-radius: 12px;
            }
            QPushButton:hover { background-color: #f5f5f5; border-color: #aaa; }
        """)
        self.btn_copy.clicked.connect(self.copy_text)
        right_layout.addWidget(self.btn_copy)

        main_layout.addWidget(left_widget, 4)
        main_layout.addWidget(right_widget, 6)
        
        self.setLayout(main_layout)

    # --- 逻辑 ---
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
            QPushButton {
                background-color: #f0fff4;
                border: 2px solid #2ecc71;
                border-radius: 20px;
                color: #27ae60;
                font-weight: bold;
            }
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
        # 注意：这里不立即启动 fake_progress，由 worker 决定

        self.worker = WorkThread(self.video_path, self.selected_model)
        self.worker.status_signal.connect(self.lbl_status.setText) 
        self.worker.progress_signal.connect(self.update_progress_ui) 
        self.worker.result_signal.connect(self.on_success)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def update_fake_progress(self):
        self.btn_start.increment_fake_progress(0.2)

    def update_progress_ui(self, val, text_override):
        self.btn_start.set_progress(val, text_override)
        # 只有当下载阶段结束 (进度>=20) 且没有文字覆盖(下载信息)时，才启动心跳
        if val >= 20 and text_override is None and not self.fake_progress_timer.isActive():
            self.fake_progress_timer.start(100)

    def on_success(self, text):
        self.fake_progress_timer.stop()
        self.btn_start.set_progress(100)
        self.text_area.setPlainText(text)
        self.reset_ui()
        QMessageBox.information(self, "成功", "转换完成！")

    def on_error(self, msg):
        self.fake_progress_timer.stop()
        self.reset_ui()
        self.lbl_status.setText("❌ 发生错误")
        QMessageBox.warning(self, "错误", msg)

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
        if self.fake_progress_timer.isActive():
            self.fake_progress_timer.stop()
        # 强制退出，不给任何弹窗机会
        os._exit(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())