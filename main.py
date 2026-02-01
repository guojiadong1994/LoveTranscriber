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
    QButtonGroup
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRectF
from PyQt6.QtGui import QFont, QColor, QPainter, QPainterPath, QIcon

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

# 🔄 顺序调整：推荐 -> 深度 -> 省电 -> 极速
MODEL_OPTIONS = [
    {"name": "🌟 推荐模式", "desc": "均衡首选", "code": "medium", "color": "#2ecc71"},
    {"name": "🧠 深度模式", "desc": "最准但慢", "code": "large-v3", "color": "#00cec9"},
    {"name": "⚡ 省电模式", "desc": "轻量快速", "code": "small", "color": "#1abc9c"},
    {"name": "🚀 极速模式", "desc": "飞一般的快", "code": "base", "color": "#3498db"}
]

# ==============================================================================
# 🎨 UI 组件：精致化
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
        # 增加平滑过渡逻辑
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

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#f0f0f0"))
        painter.drawRoundedRect(rectf, 25, 25)

        if self._progress > 0:
            prog_width = max(30, (rect.width() * (self._progress / 100.0)))
            path = QPainterPath()
            path.addRoundedRect(rectf, 25, 25)
            painter.setClipPath(path)
            painter.setBrush(QColor("#0078d7"))
            painter.drawRect(0, 0, int(prog_width), int(rect.height()))
            painter.setClipping(False)

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
        self.setFixedHeight(85)

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

class ToggleButton(QPushButton):
    """胶囊切换按钮（替代单选框）"""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setFixedHeight(35)
        self.setFont(QFont(UI_FONT, 10))
        self.update_style(False)

    def update_style(self, checked):
        if checked:
            # 激活状态：蓝色背景，白字
            self.setStyleSheet("""
                QPushButton {
                    background-color: #0078d7;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 5px 15px;
                    font-weight: bold;
                }
            """)
        else:
            # 未激活状态：灰色背景，黑字
            self.setStyleSheet("""
                QPushButton {
                    background-color: #e0e0e0;
                    color: #555;
                    border: none;
                    border-radius: 6px;
                    padding: 5px 15px;
                }
                QPushButton:hover { background-color: #d0d0d0; }
            """)

# ==============================================================================
# ✅ 核心逻辑线程 (慢速优雅进度条)
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
            self.progress_signal.emit(1) # 从 1% 开始，不突兀
            
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
            cmd_wh = [whisper_cli, "-m", model_path, "-f", tmp_wav, "-l", "zh", "-otxt", "-of", out_prefix]

            self.proc = subprocess.Popen(
                cmd_wh,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=os.path.dirname(whisper_cli),
                text=True, encoding="utf-8", errors="replace",
                startupinfo=startupinfo, creationflags=subprocess.CREATE_NO_WINDOW if platform.system()=='Windows' else 0
            )

            # 🚀 优化后的进度条算法：更慢，更均匀
            current_prog = 5.0
            target_prog = 99.0
            
            while True:
                if self.proc.poll() is not None: break
                if not self.is_running: self.proc.kill(); return
                
                # 每次只走剩余路程的 2% (之前是 5%)，步子迈小一点
                remaining = target_prog - current_prog
                step = remaining * 0.02 
                if step < 0.05: step = 0.05 # 保持极微小的蠕动
                
                current_prog += step
                self.progress_signal.emit(int(current_prog))
                
                self.proc.stdout.readline()
                time.sleep(0.1) # 刷新频率快一点，但步长小，视觉更丝滑

            if self.proc.returncode != 0: raise Exception("识别意外中断")
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
# ✅ 主窗口 (完美布局 + 胶囊切换)
# ==============================================================================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("❤️ 专属语音转文字助手")
        self.resize(1000, 650)
        self.setAcceptDrops(True)
        self.media_path = ""
        self.selected_model = "medium"
        self.full_raw_text = ""
        self.model_btns = []
        self.worker = None 
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(25)

        # === 左侧控制区 ===
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(15)

        # 1. 标题
        title = QLabel("步骤 1: 选择配置")
        title.setFont(QFont(UI_FONT, 12, QFont.Weight.Bold))
        title.setStyleSheet("color: #555;")
        left_layout.addWidget(title)

        # 2. 导入按钮
        self.btn_import = QPushButton("\n📂 点击选择 / 拖入视频\n")
        self.btn_import.setFont(QFont(UI_FONT, 14))
        self.btn_import.setFixedHeight(120)
        self.btn_import.setStyleSheet("""
            QPushButton { border: 2px dashed #aaa; border-radius: 15px; background-color: #f9f9f9; color: #555; }
            QPushButton:hover { border-color: #0078d7; background-color: #f0f8ff; color: #0078d7; }
        """)
        self.btn_import.clicked.connect(self.sel_media)
        left_layout.addWidget(self.btn_import)

        # 🔧 布局核心：这里加一点弹簧，把模型区域稍微往下压
        left_layout.addStretch(1) 

        # 3. 模型选择 (Grid)
        grid = QGridLayout()
        grid.setSpacing(10)
        for i, m in enumerate(MODEL_OPTIONS):
            b = ModelCard(m["name"], m["desc"], m["code"], m["color"])
            b.clicked.connect(lambda c, x=b: self.on_model_click(x))
            grid.addWidget(b, i // 2, i % 2)
            self.model_btns.append(b)
        left_layout.addLayout(grid)
        self.on_model_click(self.model_btns[0])

        # 4. 状态文字
        self.lbl_stat = QLabel("等待任务...")
        self.lbl_stat.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_stat.setStyleSheet("color: #666; margin-top: 10px;") # 加点上边距
        left_layout.addWidget(self.lbl_stat)

        # 🔧 布局核心：这里加更大的弹簧，把开始按钮推到底部
        left_layout.addStretch(3)

        # 5. 开始按钮
        self.btn_start = ProgressButton("✨ 开始转换")
        self.btn_start.setFixedHeight(55)
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start)
        left_layout.addWidget(self.btn_start)

        # === 右侧结果区 ===
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(10)

        # 头部布局：标题 + 胶囊切换器
        r_head_layout = QHBoxLayout()
        r_title = QLabel("步骤 2: 获取结果")
        r_title.setFont(QFont(UI_FONT, 12, QFont.Weight.Bold))
        r_title.setStyleSheet("color: #555;")
        r_head_layout.addWidget(r_title)
        
        r_head_layout.addStretch() # 把切换器推到右边

        # ✨ 新设计：胶囊切换按钮
        self.toggle_group = QButtonGroup(self)
        self.btn_mode_lines = ToggleButton("📝 分行显示")
        self.btn_mode_full = ToggleButton("📜 逗号连句")
        
        self.toggle_group.addButton(self.btn_mode_lines)
        self.toggle_group.addButton(self.btn_mode_full)
        
        # 默认选中第一个
        self.btn_mode_lines.setChecked(True)
        self.btn_mode_lines.update_style(True)
        self.btn_mode_full.update_style(False)
        
        # 绑定点击事件 (样式切换 + 功能切换)
        self.toggle_group.buttonClicked.connect(self.on_format_change)

        r_head_layout.addWidget(self.btn_mode_lines)
        r_head_layout.addWidget(self.btn_mode_full)
        
        right_layout.addLayout(r_head_layout)

        # 文本框 (自带滚动条)
        self.txt = QTextEdit()
        self.txt.setPlaceholderText("这里会显示转换结果...\n\n(支持超长文本，右侧会自动出现滚动条)")
        self.txt.setFont(QFont(UI_FONT, 12))
        self.txt.setStyleSheet("border: 1px solid #ddd; border-radius: 10px; padding: 10px; background-color: #fff;")
        right_layout.addWidget(self.txt)

        # 复制按钮
        btn_copy = QPushButton("📋 一键复制结果")
        btn_copy.setFixedHeight(55)
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

    def on_format_change(self, btn):
        # 刷新按钮样式
        self.btn_mode_lines.update_style(self.btn_mode_lines.isChecked())
        self.btn_mode_full.update_style(self.btn_mode_full.isChecked())
        # 刷新文本内容
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
        self.btn_import.setText(f"\n✅ 已就绪:\n{os.path.basename(p)}\n")
        self.btn_import.setStyleSheet(self.btn_import.styleSheet().replace("#f9f9f9", "#e8f5e9").replace("#aaa", "#2ecc71"))
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
        self.lbl_stat.setText("🎉 转换成功！")
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

    def closeEvent(self, event):
        """强制杀死后台进程"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(200)
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())