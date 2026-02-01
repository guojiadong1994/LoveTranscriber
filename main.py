import sys
import os
import time
import platform
import threading

# 界面库
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QComboBox, QTextEdit, QProgressBar,
                             QGroupBox, QMessageBox, QFileDialog, QSplitter, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QIcon, QAction

# 核心库：Faster Whisper (延迟加载，防止启动卡顿)
try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

# === 全局配置 ===
IS_MAC = (platform.system() == 'Darwin')
UI_FONT = "Microsoft YaHei" if not IS_MAC else "PingFang SC"

# === 核心工作线程 (加载+识别一体化) ===
class WorkThread(QThread):
    status_signal = pyqtSignal(str)   # 更新状态文字
    progress_signal = pyqtSignal(int) # 更新进度条 (0-100)
    result_signal = pyqtSignal(str)   # 返回结果
    error_signal = pyqtSignal(str)    # 报错

    def __init__(self, video_path, model_size):
        super().__init__()
        self.video_path = video_path
        self.model_size = model_size
        self.is_running = True

    def run(self):
        if not HAS_WHISPER:
            self.error_signal.emit("错误：未检测到 faster-whisper 库！")
            return

        try:
            # --- 第1步：加载模型 ---
            self.status_signal.emit("⏳ 第1步：正在唤醒 AI 大脑 (加载模型)...")
            self.progress_signal.emit(10)
            
            # 获取程序运行目录
            if getattr(sys, 'frozen', False):
                base_dir = os.path.dirname(sys.executable)
            else:
                base_dir = os.path.dirname(os.path.abspath(__file__))
            
            model_dir = os.path.join(base_dir, "models")
            
            # 加载模型 (自动下载/读取)
            model = WhisperModel(
                self.model_size, 
                device="cpu", 
                compute_type="int8", 
                download_root=model_dir
            )
            
            if not self.is_running: return
            self.progress_signal.emit(30)

            # --- 第2步：开始识别 ---
            self.status_signal.emit(f"🎧 第2步：正在认真听写中...\n({os.path.basename(self.video_path)})")
            
            segments, info = model.transcribe(
                self.video_path, 
                beam_size=5, 
                language="zh",
                initial_prompt="这是一段清晰的普通话，请加标点符号。"
            )

            full_text = ""
            # 这是一个估算进度的简易方法
            total_duration = info.duration
            current_time = 0

            for segment in segments:
                if not self.is_running: return
                full_text += segment.text
                current_time = segment.end
                
                # 计算进度 30% -> 95%
                if total_duration > 0:
                    progress = 30 + int((current_time / total_duration) * 65)
                    self.progress_signal.emit(min(progress, 99))

            # --- 第3步：完成 ---
            self.progress_signal.emit(100)
            self.status_signal.emit("✅ 搞定啦！请看下方结果 👇")
            self.result_signal.emit(full_text)

        except Exception as e:
            self.error_signal.emit(f"发生小意外: {str(e)}")

    def stop(self):
        self.is_running = False


# === 主界面 ===
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("❤️ 专属语音转文字助手")
        self.resize(500, 750) # 竖屏设计，像手机APP一样简单
        self.setAcceptDrops(True)
        
        self.video_path = ""
        self.worker = None
        
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(30, 40, 30, 40)

        # 1. 标题
        title = QLabel("✨ 视频转文字 ✨")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont(UI_FONT, 24, QFont.Weight.Bold))
        title.setStyleSheet("color: #333;")
        layout.addWidget(title)

        # 2. 步骤一：导入区域
        self.btn_import = QPushButton("\n📂 第一步：点击选择视频文件\n(或者把视频拖到这里)\n")
        self.btn_import.setFont(QFont(UI_FONT, 11))
        self.btn_import.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_import.setStyleSheet("""
            QPushButton {
                background-color: #f0f7ff;
                border: 2px dashed #0078d7;
                border-radius: 15px;
                color: #0078d7;
                padding: 20px;
            }
            QPushButton:hover {
                background-color: #e0efff;
            }
        """)
        self.btn_import.clicked.connect(self.select_video)
        layout.addWidget(self.btn_import)

        # 3. 步骤二：状态显示与进度
        self.status_label = QLabel("等待导入视频...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFont(QFont(UI_FONT, 10))
        self.status_label.setStyleSheet("color: #666; margin-top: 10px;")
        layout.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        self.progress.setStyleSheet("""
            QProgressBar {
                background-color: #eee;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #FF6B6B; 
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.progress)

        # 4. 步骤三：开始按钮
        self.btn_start = QPushButton("🚀 开始转换")
        self.btn_start.setFont(QFont(UI_FONT, 14, QFont.Weight.Bold))
        self.btn_start.setFixedHeight(55)
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start.setEnabled(False) # 没选文件不能点
        # 按钮样式：平时灰色，激活后粉色
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #ccc;
                color: white;
                border-radius: 27px;
                border: none;
            }
            QPushButton:enabled {
                background-color: #FF6B6B; 
                box-shadow: 0px 4px 10px rgba(255, 107, 107, 0.3);
            }
            QPushButton:enabled:hover {
                background-color: #ff5252;
            }
            QPushButton:pressed {
                background-color: #e04040;
                margin-top: 2px;
            }
        """)
        self.btn_start.clicked.connect(self.start_process)
        layout.addWidget(self.btn_start)

        # 5. 分割线
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("color: #eee;")
        layout.addWidget(line)

        # 6. 结果区域
        res_label = QLabel("📝 转换结果 (可以直接修改哦):")
        res_label.setFont(QFont(UI_FONT, 10, QFont.Weight.Bold))
        layout.addWidget(res_label)

        self.text_area = QTextEdit()
        self.text_area.setFont(QFont(UI_FONT, 11))
        self.text_area.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 10px;
                background-color: #fafafa;
                selection-background-color: #FF6B6B;
            }
            QTextEdit:focus {
                border: 1px solid #FF6B6B;
                background-color: #fff;
            }
        """)
        self.text_area.setPlaceholderText("转换后的文字会出现在这里...")
        layout.addWidget(self.text_area)

        # 7. 复制按钮
        self.btn_copy = QPushButton("📋 复制全部内容")
        self.btn_copy.setFixedHeight(45)
        self.btn_copy.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_copy.setStyleSheet("""
            QPushButton {
                background-color: #fff;
                color: #333;
                border: 1px solid #ddd;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #f5f5f5;
                border-color: #aaa;
            }
        """)
        self.btn_copy.clicked.connect(self.copy_text)
        layout.addWidget(self.btn_copy)

        self.setLayout(layout)

    # --- 逻辑功能 ---

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        file_path = e.mimeData().urls()[0].toLocalFile()
        self.load_video(file_path)

    def select_video(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频/音频 (*.mp4 *.mov *.avi *.mp3 *.m4a *.wav)")
        if f:
            self.load_video(f)

    def load_video(self, path):
        self.video_path = path
        # 更新按钮文字，显示文件名
        name = os.path.basename(path)
        self.btn_import.setText(f"\n📄 已选择：\n{name}\n")
        self.btn_import.setStyleSheet("""
            QPushButton {
                background-color: #f0fff4;
                border: 2px solid #48c774;
                border-radius: 15px;
                color: #2f855a;
            }
        """)
        self.status_label.setText("准备就绪，请点击“开始转换”")
        self.btn_start.setEnabled(True)
        self.progress.setValue(0)

    def start_process(self):
        if not self.video_path: return

        # 锁定界面
        self.btn_start.setEnabled(False)
        self.btn_import.setEnabled(False)
        self.btn_start.setText("⏳ 正在处理中...")
        self.text_area.clear()

        # 启动线程
        # 默认使用 medium 模型，精准且速度适中
        self.worker = WorkThread(self.video_path, "medium")
        self.worker.status_signal.connect(self.update_status)
        self.worker.progress_signal.connect(self.progress.setValue)
        self.worker.result_signal.connect(self.on_success)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def update_status(self, msg):
        self.status_label.setText(msg)

    def on_success(self, text):
        self.text_area.setPlainText(text)
        self.reset_ui_state()
        QMessageBox.information(self, "成功", "转换完成啦！\n快去看看结果对不对~")

    def on_error(self, msg):
        self.reset_ui_state()
        self.status_label.setText("❌ 出错啦")
        QMessageBox.warning(self, "哎呀", msg)

    def reset_ui_state(self):
        self.btn_start.setText("🚀 重新开始")
        self.btn_start.setEnabled(True)
        self.btn_import.setEnabled(True)

    def copy_text(self):
        content = self.text_area.toPlainText()
        if not content:
            self.status_label.setText("⚠️ 还没有内容可以复制哦")
            return
        QApplication.clipboard().setText(content)
        self.btn_copy.setText("✅ 已复制！")
        QTimer.singleShot(2000, lambda: self.btn_copy.setText("📋 复制全部内容"))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())