import sys
import os
import platform
import time
import subprocess
import tempfile
import traceback

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTextEdit, QMessageBox, QFileDialog, QGridLayout, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRectF
from PyQt6.QtGui import QFont, QColor, QPainter, QPainterPath, QIcon

# ==============================================================================
# 🛡️ 1. 日志配置 (保留 crash.log 以防万一，但静默运行)
# ==============================================================================
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_FILE = os.path.join(BASE_DIR, "crash.log")

import faulthandler
try:
    # 只有崩溃时才写入文件，平时静默
    log_fs = open(LOG_FILE, "w", encoding="utf-8", buffering=1)
    sys.stdout = log_fs
    sys.stderr = log_fs
    faulthandler.enable(file=log_fs, all_threads=True)
except:
    pass

# ==============================================================================
# ✅ 全局配置
# ==============================================================================
IS_MAC = (platform.system() == 'Darwin')
UI_FONT = "Microsoft YaHei" if not IS_MAC else "PingFang SC"

# 模型文件映射 (确保你有对应的 .bin 文件)
MODEL_FILE_MAP = {
    "medium": "ggml-medium.bin",
    "base": "ggml-base.bin",
    "large-v3": "ggml-large-v3.bin",
    "small": "ggml-small.bin",
}

MODEL_OPTIONS = [
    {"name": "🌟 推荐模式", "desc": "精准与速度平衡", "code": "medium", "color": "#2ecc71"},
    {"name": "🚀 极速模式", "desc": "速度最快", "code": "base", "color": "#3498db"},
    {"name": "🧠 深度模式", "desc": "超准但模型很大", "code": "large-v3", "color": "#00cec9"},
    {"name": "⚡ 省电模式", "desc": "轻量级", "code": "small", "color": "#1abc9c"}
]

# ==============================================================================
# 🎨 UI 组件 (高颜值回归)
# ==============================================================================

class ProgressButton(QPushButton):
    """带进度条动画的按钮"""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self._progress = 0.0
        self._is_processing = False
        self.default_text = text
        self.format_str = "处理中 {0}%"
        self._custom_text = None
        # 更加圆润现代的样式
        self.setStyleSheet("""
            QPushButton { 
                background-color: #0078d7; 
                color: white; 
                border-radius: 25px; 
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

        # 绘制背景槽
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#f0f0f0"))
        painter.drawRoundedRect(rectf, 25, 25)

        # 绘制进度条
        if self._progress > 0:
            prog_width = max(30, (rect.width() * (self._progress / 100.0)))
            path = QPainterPath()
            path.addRoundedRect(rectf, 25, 25)
            painter.setClipPath(path)
            painter.setBrush(QColor("#0078d7"))
            painter.drawRect(0, 0, int(prog_width), int(rect.height()))
            painter.setClipping(False)

        # 绘制文字
        painter.setPen(QColor("#333") if self._progress < 55 else QColor("white"))
        font = self.font()
        font.setPointSize(16)
        painter.setFont(font)
        txt = self._custom_text if self._custom_text else self.format_str.format(int(self._progress))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, txt)


class ModelCard(QPushButton):
    """卡片式模型选择按钮"""
    def __init__(self, title, desc, code, color, parent=None):
        super().__init__(parent)
        self.code = code
        self.default_color = color
        self.setCheckable(True)
        self.setFixedHeight(90)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        
        l1 = QLabel(title)
        l1.setFont(QFont(UI_FONT, 14, QFont.Weight.Bold))
        l1.setStyleSheet("border: none; background: transparent;")
        layout.addWidget(l1)

        l2 = QLabel(desc)
        l2.setFont(QFont(UI_FONT, 11))
        l2.setStyleSheet("color: #666; border: none; background: transparent;")
        layout.addWidget(l2)

        self.update_style(False)

    def update_style(self, s):
        if s:
            # 选中状态：带颜色边框和浅色背景
            self.setStyleSheet(
                f"QPushButton {{ background-color: {self.default_color}15; "
                f"border: 2px solid {self.default_color}; border-radius: 12px; }}"
            )
        else:
            # 未选中状态：灰色边框
            self.setStyleSheet(
                "QPushButton { background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 12px; }"
                "QPushButton:hover { border: 1px solid #bbb; background-color: #fcfcfc; }"
            )

# ==============================================================================
# ✅ 核心逻辑线程 (whisper.cpp + ffmpeg)
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

    def run(self):
        try:
            # 1. 路径检查
            ffmpeg = os.path.join(BASE_DIR, "tools", "ffmpeg", "ffmpeg.exe")
            whisper_cli = os.path.join(BASE_DIR, "tools", "whisper", "whisper-cli.exe")
            model_file = MODEL_FILE_MAP.get(self.model_code, "ggml-base.bin")
            model_path = os.path.join(BASE_DIR, "tools", "whisper", model_file)

            if not os.path.exists(ffmpeg): raise Exception("缺少 tools/ffmpeg/ffmpeg.exe")
            if not os.path.exists(whisper_cli): raise Exception("缺少 tools/whisper/whisper-cli.exe")
            if not os.path.exists(model_path): raise Exception(f"缺少模型文件：{model_file}")

            # 准备隐藏黑框的参数 (Windows专用)
            startupinfo = None
            if platform.system() == 'Windows':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

            # -----------------------------
            # 2. 抽取音频 (FFMPEG)
            # -----------------------------
            self.status_signal.emit("⏳ 正在提取音频...")
            self.progress_signal.emit(5)
            
            tmp_wav = os.path.join(tempfile.gettempdir(), f"love_{int(time.time())}.wav")
            
            # -vn:去视频 -ac 1:单声道 -ar 16000:采样率
            cmd_ff = [ffmpeg, "-y", "-i", self.media_path, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", tmp_wav]
            
            # 运行且不弹窗
            subprocess.run(
                cmd_ff, 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                startupinfo=startupinfo, creationflags=subprocess.CREATE_NO_WINDOW if platform.system()=='Windows' else 0
            )

            if not os.path.exists(tmp_wav):
                raise Exception("音频提取失败，请检查视频文件是否正常。")

            if not self.is_running: return

            # -----------------------------
            # 3. 识别 (Whisper.cpp)
            # -----------------------------
            self.status_signal.emit("🧠 正在AI思考中...")
            self.progress_signal.emit(15)

            out_prefix = os.path.join(tempfile.gettempdir(), f"love_out_{int(time.time())}")
            out_txt = out_prefix + ".txt"

            # -l zh:中文 -otxt:输出txt
            cmd_wh = [whisper_cli, "-m", model_path, "-f", tmp_wav, "-l", "zh", "-otxt", "-of", out_prefix]

            proc = subprocess.Popen(
                cmd_wh,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=os.path.dirname(whisper_cli), # 关键：在exe目录运行以找到dll
                text=True, encoding="utf-8", errors="replace",
                startupinfo=startupinfo, creationflags=subprocess.CREATE_NO_WINDOW if platform.system()=='Windows' else 0
            )

            # 模拟进度 (因为whisper-cli标准输出解析比较复杂，我们用时间估算+心跳)
            prog = 15
            while True:
                if proc.poll() is not None: break
                if not self.is_running: proc.kill(); return
                
                # 读取一行日志(虽然不显示，但可以用来判断活跃)
                line = proc.stdout.readline()
                
                if prog < 98:
                    prog += 0.5 # 慢速增加
                    self.progress_signal.emit(int(prog))
                time.sleep(0.1)

            if proc.returncode != 0:
                raise Exception("识别过程意外中断")

            if not os.path.exists(out_txt):
                raise Exception("未生成结果文件")

            with open(out_txt, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()

            # 清理
            try: os.remove(tmp_wav); os.remove(out_txt)
            except: pass

            self.progress_signal.emit(100)
            self.status_signal.emit("✅ 完成！")
            self.result_signal.emit(text)

        except Exception as e:
            traceback.print_exc()
            self.error_signal.emit(str(e))

# ==============================================================================
# ✅ 主窗口 (精致版)
# ==============================================================================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("❤️ 专属语音转文字助手 (最终珍藏版)")
        self.resize(1000, 650)
        self.setAcceptDrops(True)
        self.media_path = ""
        self.selected_model = "medium"
        self.model_btns = []
        self.init_ui()

    def init_ui(self):
        # 整体左右布局
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # === 左侧控制区 (40%) ===
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(15)

        # 1. 标题
        title = QLabel("步骤 1: 选择配置")
        title.setFont(QFont(UI_FONT, 12, QFont.Weight.Bold))
        title.setStyleSheet("color: #555;")
        left_layout.addWidget(title)

        # 2. 导入按钮
        self.btn_import = QPushButton("\n📂 点击选择 / 拖入视频文件\n")
        self.btn_import.setFont(QFont(UI_FONT, 14))
        self.btn_import.setFixedHeight(120)
        self.btn_import.setStyleSheet("""
            QPushButton { 
                border: 2px dashed #aaa; 
                border-radius: 15px; 
                background-color: #f9f9f9; 
                color: #555; 
            }
            QPushButton:hover { border-color: #0078d7; background-color: #f0f8ff; color: #0078d7; }
        """)
        self.btn_import.clicked.connect(self.sel_media)
        left_layout.addWidget(self.btn_import)

        # 3. 模型选择
        grid = QGridLayout()
        grid.setSpacing(10)
        for i, m in enumerate(MODEL_OPTIONS):
            b = ModelCard(m["name"], m["desc"], m["code"], m["color"])
            b.clicked.connect(lambda c, x=b: self.on_model_click(x))
            grid.addWidget(b, i // 2, i % 2)
            self.model_btns.append(b)
        left_layout.addLayout(grid)
        self.on_model_click(self.model_btns[0]) # 默认选中第一个

        # 4. 状态与开始
        self.lbl_stat = QLabel("等待任务...")
        self.lbl_stat.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_stat.setStyleSheet("color: #666; font-size: 14px;")
        left_layout.addWidget(self.lbl_stat)

        self.btn_start = ProgressButton("✨ 开始转换")
        self.btn_start.setFixedHeight(50)
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start)
        left_layout.addWidget(self.btn_start)

        left_layout.addStretch() # 底部弹簧

        # === 右侧结果区 (60%) ===
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(10)

        r_title = QLabel("步骤 2: 获取结果")
        r_title.setFont(QFont(UI_FONT, 12, QFont.Weight.Bold))
        r_title.setStyleSheet("color: #555;")
        right_layout.addWidget(r_title)

        self.txt = QTextEdit()
        self.txt.setPlaceholderText("转换后的文字将显示在这里...\n\n(转换过程中请勿关闭软件)")
        self.txt.setFont(QFont(UI_FONT, 12))
        self.txt.setStyleSheet("border: 1px solid #ddd; border-radius: 10px; padding: 10px; background-color: #fff;")
        right_layout.addWidget(self.txt)

        btn_copy = QPushButton("📋 一键复制结果")
        btn_copy.setFixedHeight(45)
        btn_copy.setFont(QFont(UI_FONT, 12))
        btn_copy.setStyleSheet("""
            QPushButton { background-color: #2ecc71; color: white; border-radius: 10px; border: none; font-weight: bold; }
            QPushButton:hover { background-color: #27ae60; }
        """)
        btn_copy.clicked.connect(self.copy_result)
        right_layout.addWidget(btn_copy)

        # 添加到主布局
        main_layout.addWidget(left_widget, 4)
        main_layout.addWidget(right_widget, 6)
        self.setLayout(main_layout)

        # 设置整体背景
        self.setStyleSheet("background-color: #fcfcfc;")

    def on_model_click(self, b):
        for x in self.model_btns:
            x.setChecked(x == b)
            x.update_style(x == b)
        self.selected_model = b.code

    def dragEnterEvent(self, e):
        e.accept() if e.mimeData().hasUrls() else e.ignore()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls: self.load(urls[0].toLocalFile())

    def sel_media(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "Media (*.mp4 *.mov *.avi *.mkv *.mp3 *.wav *.m4a)")
        if f: self.load(f)

    def load(self, p):
        self.media_path = p
        self.btn_import.setText(f"\n✅ 已就绪:\n{os.path.basename(p)}\n(再次点击可更换)")
        self.btn_import.setStyleSheet(self.btn_import.styleSheet().replace("#f9f9f9", "#e8f5e9").replace("#aaa", "#2ecc71"))
        self.btn_start.setEnabled(True)
        self.lbl_stat.setText("准备就绪，点击开始")

    def start(self):
        self.btn_start.start_processing()
        self.btn_import.setEnabled(False)
        self.txt.clear()
        
        self.worker = TranscribeThread(self.media_path, self.selected_model)
        self.worker.status_signal.connect(self.lbl_stat.setText)
        self.worker.progress_signal.connect(self.btn_start.set_progress)
        self.worker.result_signal.connect(self.done)
        self.worker.error_signal.connect(self.fail)
        self.worker.start()

    def done(self, text):
        self.txt.setPlainText(text)
        self.lbl_stat.setText("🎉 转换成功！")
        self.reset_ui()

    def fail(self, err):
        self.lbl_stat.setText("❌ 出错")
        self.txt.setPlainText(f"发生错误:\n{err}\n\n请检查 tools 文件夹是否完整。")
        self.reset_ui()
        QMessageBox.warning(self, "抱歉", f"出现了一些问题：\n{err}")

    def reset_ui(self):
        self.btn_start.stop_processing()
        self.btn_import.setEnabled(True)

    def copy_result(self):
        self.txt.selectAll()
        self.txt.copy()
        self.lbl_stat.setText("已复制到剪贴板！")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())