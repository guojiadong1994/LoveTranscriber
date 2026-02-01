import sys
import os
import platform
import time
import subprocess
import tempfile
import traceback
import threading 

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTextEdit, QMessageBox, QFileDialog, QGridLayout, 
    QButtonGroup, QSizePolicy, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRectF
from PyQt6.QtGui import QFont, QColor, QPainter, QPainterPath

# ==============================================================================
# ✅ 全局配置
# ==============================================================================
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IS_MAC = (platform.system() == 'Darwin')
UI_FONT = "Microsoft YaHei" if not IS_MAC else "PingFang SC"

MODEL_FILE_MAP = {
    "medium": "ggml-medium.bin",
    "base": "ggml-base.bin",
    "large-v3": "ggml-large-v3.bin",
    "small": "ggml-small.bin",
}

MODEL_OPTIONS = [
    {"name": "🌟 推荐模式", "desc": "均衡首选", "code": "medium", "color": "#2ecc71"},
    {"name": "🧠 深度模式", "desc": "最准但慢", "code": "large-v3", "color": "#00cec9"},
    {"name": "⚡ 省电模式", "desc": "轻量快速", "code": "small", "color": "#1abc9c"},
    {"name": "🚀 极速模式", "desc": "飞一般的快", "code": "base", "color": "#3498db"}
]

# ==============================================================================
# 🎨 UI 组件
# ==============================================================================

class ProgressButton(QPushButton):
    """带进度条动画的按钮"""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self._progress = 0.0
        self._is_processing = False
        self.default_text = text
        self.format_str = "正在转换... {0}%"
        self._custom_text = None
        self.setStyleSheet("""
            QPushButton { 
                background-color: #0078d7; 
                color: white; 
                border-radius: 22px; 
                font-weight: bold; 
                font-size: 18px; 
                border: none;
            }
            QPushButton:hover { background-color: #0063b1; }
            QPushButton:pressed { background-color: #005a9e; }
            QPushButton:disabled { background-color: #e0e0e0; color: #999; }
        """)

    def set_progress(self, value):
        self._progress = float(value)
        self.update()

    def set_text_override(self, text):
        self._custom_text = text
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
        if not self._is_processing:
            super().paintEvent(event)
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        rectf = QRectF(rect)

        # 背景
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#f0f0f0"))
        painter.drawRoundedRect(rectf, 22, 22)

        # 进度条
        if self._progress > 0:
            prog_width = max(30, (rect.width() * (self._progress / 100.0)))
            if prog_width > rect.width(): prog_width = rect.width()
            
            path = QPainterPath()
            path.addRoundedRect(rectf, 22, 22)
            painter.setClipPath(path)
            painter.setBrush(QColor("#0078d7"))
            painter.drawRect(0, 0, int(prog_width), int(rect.height()))
            painter.setClipping(False)

        # 文字
        painter.setPen(QColor("#333") if self._progress < 55 else QColor("white"))
        font = self.font()
        font.setPointSize(16)
        painter.setFont(font)
        txt = self._custom_text if self._custom_text else self.format_str.format(int(self._progress))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, txt)


class ModelCard(QPushButton):
    def __init__(self, title, desc, code, color, parent=None):
        super().__init__(parent)
        self.code = code
        self.default_color = color
        self.setCheckable(True)
        self.setFixedHeight(80) 

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        l1 = QLabel(title)
        l1.setFont(QFont(UI_FONT, 12, QFont.Weight.Bold)) 
        l1.setStyleSheet("border: none; background: transparent;")
        layout.addWidget(l1)

        l2 = QLabel(desc)
        l2.setFont(QFont(UI_FONT, 9))
        l2.setStyleSheet("color: #666; border: none; background: transparent;")
        layout.addWidget(l2)

        self.update_style(False)

    def update_style(self, s):
        if s:
            self.setStyleSheet(
                f"QPushButton {{ background-color: {self.default_color}15; "
                f"border: 2px solid {self.default_color}; border-radius: 12px; }}"
            )
        else:
            self.setStyleSheet(
                "QPushButton { background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 12px; }"
                "QPushButton:hover { border: 1px solid #bbb; background-color: #fcfcfc; }"
            )

class ToggleButton(QPushButton):
    """胶囊切换按钮"""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setFixedHeight(45) 
        self.setFont(QFont(UI_FONT, 11))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.update_style(False)

    def update_style(self, checked):
        if checked:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #0078d7;
                    color: white;
                    border: none;
                    border-radius: 10px;
                    font-weight: bold;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #f0f0f0;
                    color: #666;
                    border: 1px solid #ddd;
                    border-radius: 10px;
                }
                QPushButton:hover { background-color: #e8e8e8; }
            """)

# ==============================================================================
# ✅ 核心逻辑线程 (极简稳定版)
# ==============================================================================
class TranscribeThread(QThread):
    status_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, media_path, model_code):
        super().__init__()
        self.media_path = media_path
        self.model_code = model_code
        self.is_running = True
        self.proc = None 

    def stop(self):
        self.is_running = False
        if self.proc:
            try: self.proc.kill()
            except: pass

    def _drain_stdout(self, pipe):
        try:
            for _ in pipe: pass 
        except: pass

    def run(self):
        try:
            ffmpeg = os.path.join(BASE_DIR, "tools", "ffmpeg", "ffmpeg.exe")
            whisper_cli = os.path.join(BASE_DIR, "tools", "whisper", "whisper-cli.exe")
            model_file = MODEL_FILE_MAP.get(self.model_code, "ggml-base.bin")
            model_path = os.path.join(BASE_DIR, "tools", "whisper", model_file)

            if not os.path.exists(ffmpeg): raise Exception("缺少 ffmpeg.exe")
            if not os.path.exists(whisper_cli): raise Exception("缺少 whisper-cli.exe")
            if not os.path.exists(model_path): raise Exception(f"缺少模型：{model_file}")

            startupinfo = None
            if platform.system() == 'Windows':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

            # --- 1. 抽取音频 ---
            self.status_signal.emit("⏳ 正在提取音频...")
            self.progress_signal.emit(5)
            
            tmp_wav = os.path.join(tempfile.gettempdir(), f"love_{int(time.time())}.wav")
            cmd_ff = [ffmpeg, "-y", "-i", self.media_path, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", tmp_wav]
            
            subprocess.run(
                cmd_ff, 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                startupinfo=startupinfo, creationflags=subprocess.CREATE_NO_WINDOW if platform.system()=='Windows' else 0
            )

            if not os.path.exists(tmp_wav): raise Exception("音频提取失败")
            if not self.is_running: return

            # --- 2. 识别 ---
            self.status_signal.emit("🧠 正在AI思考中...")
            
            out_prefix = os.path.join(tempfile.gettempdir(), f"love_out_{int(time.time())}")
            out_txt = out_prefix + ".txt"
            
            # 🔥 核心修改：去掉了 -p 参数，防止报错！
            cmd_wh = [
                whisper_cli, "-m", model_path, "-f", tmp_wav, 
                "-l", "zh", # 依然保留强制中文
                "-otxt", "-of", out_prefix
            ]

            self.proc = subprocess.Popen(
                cmd_wh,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=os.path.dirname(whisper_cli),
                text=True, encoding="utf-8", errors="replace",
                startupinfo=startupinfo, creationflags=subprocess.CREATE_NO_WINDOW if platform.system()=='Windows' else 0
            )

            t = threading.Thread(target=self._drain_stdout, args=(self.proc.stdout,))
            t.daemon = True
            t.start()

            # 🚀 进度条：超级慢速匀速版
            current_prog = 5.0
            
            while True:
                if self.proc.poll() is not None:
                    break
                
                if not self.is_running: 
                    self.proc.kill()
                    return
                
                if current_prog < 99.0:
                    # 🔥 修改点：步长从 1.5 改为 0.5 (慢了 3 倍)
                    # 这样进度条会走得非常稳，不会一下跑完
                    current_prog += 0.5 
                    self.progress_signal.emit(int(current_prog))
                
                time.sleep(0.1) 

            if self.proc.returncode != 0: 
                if not os.path.exists(out_txt):
                    raise Exception("识别意外中断，未生成结果")

            if not os.path.exists(out_txt): raise Exception("未生成结果")

            with open(out_txt, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()

            try: os.remove(tmp_wav); os.remove(out_txt)
            except: pass

            self.progress_signal.emit(100) 
            self.status_signal.emit("✅ 转换完成")
            self.result_signal.emit(text)

        except Exception as e:
            self.error_signal.emit(str(e))

# ==============================================================================
# ✅ 主窗口 (完美对齐版)
# ==============================================================================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("❤️ 专属语音转文字助手")
        self.resize(950, 600) 
        self.setAcceptDrops(True)
        self.media_path = ""
        self.selected_model = "medium"
        self.full_raw_text = ""
        self.model_btns = []
        self.worker = None 
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # === 左侧控制区 (40%) ===
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10) 
        left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 1. 导入
        title1 = QLabel("步骤 1: 选择视频") 
        title1.setFont(QFont(UI_FONT, 11, QFont.Weight.Bold))
        title1.setStyleSheet("color: #444;")
        left_layout.addWidget(title1)

        self.btn_import = QPushButton("\n📂 点击 / 拖入文件\n")
        self.btn_import.setFont(QFont(UI_FONT, 13))
        self.btn_import.setFixedHeight(100) 
        self.btn_import.setStyleSheet("""
            QPushButton { border: 2px dashed #aaa; border-radius: 12px; background-color: #fcfcfc; color: #666; }
            QPushButton:hover { border-color: #0078d7; background-color: #f0f8ff; color: #0078d7; }
        """)
        self.btn_import.clicked.connect(self.sel_media)
        left_layout.addWidget(self.btn_import)

        left_layout.addSpacing(20) 

        # 2. 模式
        title2 = QLabel("步骤 2: 选择模式")
        title2.setFont(QFont(UI_FONT, 11, QFont.Weight.Bold))
        title2.setStyleSheet("color: #444;")
        left_layout.addWidget(title2)

        grid = QGridLayout()
        grid.setSpacing(8)
        for i, m in enumerate(MODEL_OPTIONS):
            b = ModelCard(m["name"], m["desc"], m["code"], m["color"])
            b.clicked.connect(lambda c, x=b: self.on_model_click(x))
            grid.addWidget(b, i // 2, i % 2)
            self.model_btns.append(b)
        left_layout.addLayout(grid)
        self.on_model_click(self.model_btns[0])

        left_layout.addSpacing(15)

        # 3. 状态与开始
        self.lbl_stat = QLabel("准备就绪")
        self.lbl_stat.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_stat.setStyleSheet("color: #888; font-size: 13px; margin-bottom: 2px;")
        left_layout.addWidget(self.lbl_stat)

        left_layout.addStretch(1)

        self.btn_start = ProgressButton("✨ 开始转换")
        self.btn_start.setFixedHeight(50) 
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start)
        left_layout.addWidget(self.btn_start)

        # === 右侧结果区 (60%) ===
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(10) 

        # 1. 文本框
        self.txt = QTextEdit()
        self.txt.setPlaceholderText("转换结果将显示在这里...")
        self.txt.setFont(QFont(UI_FONT, 11))
        self.txt.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.txt.setStyleSheet("border: 1px solid #ddd; border-radius: 10px; padding: 10px; background-color: #fff;")
        right_layout.addWidget(self.txt)

        # 2. 底部功能区
        bottom_box = QVBoxLayout()
        bottom_box.setSpacing(10)
        bottom_box.setContentsMargins(0, 0, 0, 0) 

        toggles_layout = QHBoxLayout()
        toggles_layout.setSpacing(10)
        
        self.toggle_group = QButtonGroup(self)
        self.btn_mode_lines = ToggleButton("📝 分行显示")
        self.btn_mode_full = ToggleButton("📜 逗号连句")
        
        self.toggle_group.addButton(self.btn_mode_lines)
        self.toggle_group.addButton(self.btn_mode_full)
        self.toggle_group.buttonClicked.connect(self.on_format_change)
        
        self.btn_mode_lines.setChecked(True)
        self.btn_mode_lines.update_style(True)
        
        toggles_layout.addWidget(self.btn_mode_lines)
        toggles_layout.addWidget(self.btn_mode_full)
        
        bottom_box.addLayout(toggles_layout)

        btn_copy = QPushButton("📋 一键复制结果")
        btn_copy.setFixedHeight(50)
        btn_copy.setFont(QFont(UI_FONT, 12))
        btn_copy.setStyleSheet("""
            QPushButton { background-color: #2ecc71; color: white; border-radius: 22px; border: none; font-weight: bold; }
            QPushButton:hover { background-color: #27ae60; }
        """)
        btn_copy.clicked.connect(self.copy_result)
        bottom_box.addWidget(btn_copy)

        right_layout.addLayout(bottom_box)

        main_layout.addWidget(left_widget, 4)
        main_layout.addWidget(right_widget, 6)
        self.setLayout(main_layout)
        self.setStyleSheet("background-color: #fdfdfd;")

    def on_model_click(self, b):
        for x in self.model_btns:
            x.setChecked(x == b)
            x.update_style(x == b)
        self.selected_model = b.code

    def on_format_change(self, btn):
        self.btn_mode_lines.update_style(self.btn_mode_lines.isChecked())
        self.btn_mode_full.update_style(self.btn_mode_full.isChecked())
        self.update_text_display()

    def dragEnterEvent(self, e): e.accept() if e.mimeData().hasUrls() else e.ignore()
    def dropEvent(self, e): 
        urls = e.mimeData().urls()
        if urls: self.load(urls[0].toLocalFile())

    def sel_media(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "Media (*.mp4 *.mov *.avi *.mkv *.mp3 *.wav *.m4a)")
        if f: self.load(f)

    def load(self, p):
        self.media_path = p
        self.btn_import.setText(f"\n✅ 已加载:\n{os.path.basename(p)}\n")
        self.btn_import.setStyleSheet(self.btn_import.styleSheet().replace("#fcfcfc", "#e8f5e9").replace("#aaa", "#2ecc71"))
        self.btn_start.setEnabled(True)
        self.lbl_stat.setText("准备就绪")

    def start(self):
        self.btn_start.start_processing()
        self.btn_import.setEnabled(False)
        self.txt.clear()
        self.full_raw_text = ""
        
        self.worker = TranscribeThread(self.media_path, self.selected_model)
        self.worker.status_signal.connect(self.lbl_stat.setText)
        self.worker.progress_signal.connect(self.btn_start.set_progress)
        self.worker.result_signal.connect(self.done)
        self.worker.error_signal.connect(self.fail)
        self.worker.start()

    def done(self, text):
        self.full_raw_text = text
        self.update_text_display()
        self.lbl_stat.setText("转换完成")
        self.reset_ui()

    def update_text_display(self):
        if not self.full_raw_text: return
        if self.btn_mode_lines.isChecked():
            self.txt.setPlainText(self.full_raw_text)
        else:
            clean_text = self.full_raw_text.replace('\n', '，').replace('\r', '')
            while "，，" in clean_text: clean_text = clean_text.replace("，，", "，")
            self.txt.setPlainText(clean_text)

    def fail(self, err):
        self.lbl_stat.setText("出错")
        self.txt.setPlainText(f"错误信息:\n{err}")
        self.reset_ui()
        QMessageBox.warning(self, "出错啦", f"{err}")

    def reset_ui(self):
        self.btn_start.stop_processing()
        self.btn_import.setEnabled(True)

    def copy_result(self):
        self.txt.selectAll()
        self.txt.copy()
        self.lbl_stat.setText("已复制")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(200)
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())