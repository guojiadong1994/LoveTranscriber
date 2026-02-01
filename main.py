import sys
import os
import platform
import time
import subprocess
import tempfile
import traceback
import re

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTextEdit, QMessageBox, QFileDialog, QGridLayout, QFrame,
    QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRectF
from PyQt6.QtGui import QFont, QColor, QPainter, QPainterPath, QIcon

# ==============================================================================
# 🛡️ 日志配置
# ==============================================================================
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_FILE = os.path.join(BASE_DIR, "crash.log")

import faulthandler
try:
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
# 🎨 UI 组件
# ==============================================================================

class ProgressButton(QPushButton):
    """带丝滑进度条动画的按钮"""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self._progress = 0.0
        self._is_processing = False
        self.default_text = text
        self.format_str = "处理中 {0}%"
        self._custom_text = None
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
        # 增加平滑过渡逻辑，防止倒退
        if float(value) > self._progress:
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
        painter.drawRoundedRect(rectf, 25, 25)

        # 进度条
        if self._progress > 0:
            prog_width = max(30, (rect.width() * (self._progress / 100.0)))
            path = QPainterPath()
            path.addRoundedRect(rectf, 25, 25)
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
        self.setFixedHeight(85) # 稍微调低高度，更精致

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 8, 15, 8)
        
        l1 = QLabel(title)
        l1.setFont(QFont(UI_FONT, 13, QFont.Weight.Bold))
        l1.setStyleSheet("border: none; background: transparent;")
        layout.addWidget(l1)

        l2 = QLabel(desc)
        l2.setFont(QFont(UI_FONT, 10))
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

# ==============================================================================
# ✅ 核心逻辑线程 (进度条算法优化)
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
            ffmpeg = os.path.join(BASE_DIR, "tools", "ffmpeg", "ffmpeg.exe")
            whisper_cli = os.path.join(BASE_DIR, "tools", "whisper", "whisper-cli.exe")
            model_file = MODEL_FILE_MAP.get(self.model_code, "ggml-base.bin")
            model_path = os.path.join(BASE_DIR, "tools", "whisper", model_file)

            if not os.path.exists(ffmpeg): raise Exception("缺少 tools/ffmpeg/ffmpeg.exe")
            if not os.path.exists(whisper_cli): raise Exception("缺少 tools/whisper/whisper-cli.exe")
            if not os.path.exists(model_path): raise Exception(f"缺少模型文件：{model_file}")

            startupinfo = None
            if platform.system() == 'Windows':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

            # --- 1. 抽取音频 ---
            self.status_signal.emit("⏳ 正在准备音频...")
            self.progress_signal.emit(5) # 初始跳动
            
            tmp_wav = os.path.join(tempfile.gettempdir(), f"love_{int(time.time())}.wav")
            cmd_ff = [ffmpeg, "-y", "-i", self.media_path, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", tmp_wav]
            
            subprocess.run(
                cmd_ff, 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                startupinfo=startupinfo, creationflags=subprocess.CREATE_NO_WINDOW if platform.system()=='Windows' else 0
            )

            if not os.path.exists(tmp_wav): raise Exception("音频提取失败。")
            if not self.is_running: return

            # --- 2. 识别 (丝滑进度条逻辑) ---
            self.status_signal.emit("🧠 正在努力听写中...")
            
            out_prefix = os.path.join(tempfile.gettempdir(), f"love_out_{int(time.time())}")
            out_txt = out_prefix + ".txt"

            cmd_wh = [whisper_cli, "-m", model_path, "-f", tmp_wav, "-l", "zh", "-otxt", "-of", out_prefix]

            proc = subprocess.Popen(
                cmd_wh,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=os.path.dirname(whisper_cli),
                text=True, encoding="utf-8", errors="replace",
                startupinfo=startupinfo, creationflags=subprocess.CREATE_NO_WINDOW if platform.system()=='Windows' else 0
            )

            # 🚀 算法优化：渐近式进度条 (Zeno's Paradox)
            # 让进度条永远在动，但永远不超过 98%，直到真正结束
            current_prog = 10.0
            target_prog = 98.0
            
            while True:
                if proc.poll() is not None: break
                if not self.is_running: proc.kill(); return
                
                # 关键算法：每次只走剩下路程的一小部分
                # 这样越往后走越慢，但一直在动，不会卡死
                remaining = target_prog - current_prog
                step = remaining * 0.05  # 每次走剩余的 5%
                if step < 0.1: step = 0.1 # 保持最低动量
                
                current_prog += step
                if current_prog > 99: current_prog = 99
                
                self.progress_signal.emit(int(current_prog))
                
                # 读取输出防止缓存堵塞
                proc.stdout.readline()
                time.sleep(0.2) # 刷新频率

            if proc.returncode != 0: raise Exception("识别意外中断")
            if not os.path.exists(out_txt): raise Exception("未生成结果")

            with open(out_txt, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()

            try: os.remove(tmp_wav); os.remove(out_txt)
            except: pass

            self.progress_signal.emit(100) # 最后瞬间拉满
            self.status_signal.emit("✅ 搞定啦！")
            self.result_signal.emit(text)

        except Exception as e:
            traceback.print_exc()
            self.error_signal.emit(str(e))

# ==============================================================================
# ✅ 主窗口 (完美对齐 + 双模式输出)
# ==============================================================================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("❤️ 专属语音转文字助手 (完美版)")
        self.resize(1000, 650)
        self.setAcceptDrops(True)
        self.media_path = ""
        self.selected_model = "medium"
        self.full_raw_text = "" # 存储原始文本
        self.model_btns = []
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(25)

        # === 左侧控制区 (40%) ===
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(15)

        title = QLabel("步骤 1: 选择配置")
        title.setFont(QFont(UI_FONT, 12, QFont.Weight.Bold))
        title.setStyleSheet("color: #555;")
        left_layout.addWidget(title)

        self.btn_import = QPushButton("\n📂 点击选择 / 拖入视频\n")
        self.btn_import.setFont(QFont(UI_FONT, 14))
        self.btn_import.setFixedHeight(120)
        self.btn_import.setStyleSheet("""
            QPushButton { border: 2px dashed #aaa; border-radius: 15px; background-color: #f9f9f9; color: #555; }
            QPushButton:hover { border-color: #0078d7; background-color: #f0f8ff; color: #0078d7; }
        """)
        self.btn_import.clicked.connect(self.sel_media)
        left_layout.addWidget(self.btn_import)

        grid = QGridLayout()
        grid.setSpacing(10)
        for i, m in enumerate(MODEL_OPTIONS):
            b = ModelCard(m["name"], m["desc"], m["code"], m["color"])
            b.clicked.connect(lambda c, x=b: self.on_model_click(x))
            grid.addWidget(b, i // 2, i % 2)
            self.model_btns.append(b)
        left_layout.addLayout(grid)
        self.on_model_click(self.model_btns[0])

        self.lbl_stat = QLabel("等待任务...")
        self.lbl_stat.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_stat.setStyleSheet("color: #666;")
        left_layout.addWidget(self.lbl_stat)

        # 🔧 布局核心：添加弹簧，把“开始转换”按钮推到底部
        left_layout.addStretch() 

        self.btn_start = ProgressButton("✨ 开始转换")
        self.btn_start.setFixedHeight(55) # 和右边对齐高度
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start)
        left_layout.addWidget(self.btn_start)

        # === 右侧结果区 (60%) ===
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(10)

        # 右侧头部布局：标题 + 单选按钮
        r_head_layout = QHBoxLayout()
        r_title = QLabel("步骤 2: 获取结果")
        r_title.setFont(QFont(UI_FONT, 12, QFont.Weight.Bold))
        r_title.setStyleSheet("color: #555;")
        r_head_layout.addWidget(r_title)
        
        r_head_layout.addStretch() # 把单选按钮推到右边

        # ✨ 新增：格式选择
        self.rb_lines = QRadioButton("📝 分行显示")
        self.rb_full = QRadioButton("📜 逗号连句")
        self.rb_lines.setChecked(True) # 默认分行
        
        # 样式美化
        rb_style = "QRadioButton { font-size: 13px; color: #333; } QRadioButton::indicator { width: 16px; height: 16px; }"
        self.rb_lines.setStyleSheet(rb_style)
        self.rb_full.setStyleSheet(rb_style)
        
        # 绑定事件
        self.rb_lines.toggled.connect(self.update_text_display)
        self.rb_full.toggled.connect(self.update_text_display)
        
        r_head_layout.addWidget(self.rb_lines)
        r_head_layout.addWidget(self.rb_full)
        
        right_layout.addLayout(r_head_layout)

        self.txt = QTextEdit()
        self.txt.setPlaceholderText("转换结果将显示在这里...")
        self.txt.setFont(QFont(UI_FONT, 12))
        self.txt.setStyleSheet("border: 1px solid #ddd; border-radius: 10px; padding: 10px; background-color: #fff;")
        right_layout.addWidget(self.txt)

        btn_copy = QPushButton("📋 一键复制结果")
        btn_copy.setFixedHeight(55) # 高度与左侧“开始转换”一致，实现视觉对齐
        btn_copy.setFont(QFont(UI_FONT, 12))
        btn_copy.setStyleSheet("""
            QPushButton { background-color: #2ecc71; color: white; border-radius: 25px; border: none; font-weight: bold; }
            QPushButton:hover { background-color: #27ae60; }
        """)
        btn_copy.clicked.connect(self.copy_result)
        right_layout.addWidget(btn_copy)

        main_layout.addWidget(left_widget, 4)
        main_layout.addWidget(right_widget, 6)
        self.setLayout(main_layout)
        self.setStyleSheet("background-color: #fcfcfc;")

    def on_model_click(self, b):
        for x in self.model_btns:
            x.setChecked(x == b)
            x.update_style(x == b)
        self.selected_model = b.code

    def dragEnterEvent(self, e): e.accept() if e.mimeData().hasUrls() else e.ignore()
    def dropEvent(self, e): 
        urls = e.mimeData().urls()
        if urls: self.load(urls[0].toLocalFile())

    def sel_media(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "Media (*.mp4 *.mov *.avi *.mkv *.mp3 *.wav *.m4a)")
        if f: self.load(f)

    def load(self, p):
        self.media_path = p
        self.btn_import.setText(f"\n✅ 已就绪:\n{os.path.basename(p)}\n")
        self.btn_import.setStyleSheet(self.btn_import.styleSheet().replace("#f9f9f9", "#e8f5e9").replace("#aaa", "#2ecc71"))
        self.btn_start.setEnabled(True)
        self.lbl_stat.setText("准备就绪")

    def start(self):
        self.btn_start.start_processing()
        self.btn_import.setEnabled(False)
        self.txt.clear()
        self.full_raw_text = "" # 清空缓存
        
        self.worker = TranscribeThread(self.media_path, self.selected_model)
        self.worker.status_signal.connect(self.lbl_stat.setText)
        self.worker.progress_signal.connect(self.btn_start.set_progress)
        self.worker.result_signal.connect(self.done)
        self.worker.error_signal.connect(self.fail)
        self.worker.start()

    def done(self, text):
        self.full_raw_text = text # 保存原始文本
        self.update_text_display() # 根据当前单选按钮状态显示
        self.lbl_stat.setText("🎉 转换成功！")
        self.reset_ui()

    def update_text_display(self):
        """根据用户选择的模式刷新文本框"""
        if not self.full_raw_text: return
        
        if self.rb_lines.isChecked():
            # 模式1: 原汁原味 (保持换行)
            self.txt.setPlainText(self.full_raw_text)
        else:
            # 模式2: 逗号连句 (去除换行，变成长句)
            # 把换行符替换成中文逗号，并处理可能出现的连续逗号
            clean_text = self.full_raw_text.replace('\n', '，').replace('\r', '')
            # 简单的清理逻辑，防止出现 ",,"
            while "，，" in clean_text:
                clean_text = clean_text.replace("，，", "，")
            self.txt.setPlainText(clean_text)

    def fail(self, err):
        self.lbl_stat.setText("❌ 出错")
        self.txt.setPlainText(f"错误信息:\n{err}\n\n请确保 tools 文件夹完整。")
        self.reset_ui()
        QMessageBox.warning(self, "出错啦", f"{err}")

    def reset_ui(self):
        self.btn_start.stop_processing()
        self.btn_import.setEnabled(True)

    def copy_result(self):
        self.txt.selectAll()
        self.txt.copy()
        self.lbl_stat.setText("已复制！")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())