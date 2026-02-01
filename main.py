import sys
import os
import platform
import time
import subprocess
import tempfile
import traceback

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTextEdit, QMessageBox, QFileDialog, QGridLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPainter, QPainterPath
from PyQt6.QtCore import QRectF

# ==============================================================================
# 🛡️ 1. 目录与日志（保留你的 crash.log 习惯）
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
    print(f"===== START {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
    print("Engine: whisper.cpp (whisper-cli.exe) + ffmpeg")
except:
    pass

# ==============================================================================
# ✅ 全局配置
# ==============================================================================
IS_MAC = (platform.system() == 'Darwin')
UI_FONT = "Microsoft YaHei" if not IS_MAC else "PingFang SC"

# UI 四个模式仍保留（但现在对应 ggml 模型文件）
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
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self._progress = 0.0
        self._is_processing = False
        self.default_text = text
        self.format_str = "运行中 {0}%"
        self._custom_text = None
        self.setStyleSheet(
            "QPushButton { background-color: #0078d7; color: white; border-radius: 30px; "
            "font-weight: bold; font-size: 20px; } "
            "QPushButton:disabled { background-color: #cccccc; color: #888; }"
        )

    def set_progress(self, value):
        self._progress = float(value)
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
        rectf = QRectF(self.rect())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#f0f0f0"))
        painter.drawRoundedRect(rectf, 30, 30)

        if self._progress > 0:
            prog_width = max(30, (self.rect().width() * (self._progress / 100.0)))
            path = QPainterPath()
            path.addRoundedRect(rectf, 30, 30)
            painter.setClipPath(path)
            painter.setBrush(QColor("#0078d7"))
            painter.drawRect(0, 0, int(prog_width), int(self.rect().height()))
            painter.setClipping(False)

        painter.setPen(QColor("#333") if self._progress < 55 else QColor("white"))
        font = self.font()
        font.setPointSize(16)
        painter.setFont(font)
        txt = self._custom_text if self._custom_text else self.format_str.format(int(self._progress))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, txt)


class ModelCard(QPushButton):
    def __init__(self, title, desc, code, color, parent=None):
        super().__init__(parent)
        self.code = code
        self.default_color = color
        self.setCheckable(True)
        self.setFixedHeight(100)

        layout = QVBoxLayout(self)
        l1 = QLabel(title)
        l1.setFont(QFont(UI_FONT, 15, QFont.Weight.Bold))
        layout.addWidget(l1)

        l2 = QLabel(desc)
        l2.setFont(QFont(UI_FONT, 13))
        layout.addWidget(l2)

        self.update_style(False)

    def update_style(self, s):
        if s:
            self.setStyleSheet(
                f"QPushButton {{ background-color: {self.default_color}15; "
                f"border: 3px solid {self.default_color}; border-radius: 12px; }}"
            )
        else:
            self.setStyleSheet(
                "QPushButton { background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 12px; }"
            )

# ==============================================================================
# ✅ 离线识别线程：ffmpeg 抽音频 + whisper-cli 转写
# ==============================================================================
class TranscribeThread(QThread):
    status_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    stage_signal = pyqtSignal(str)
    result_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, media_path, model_code):
        super().__init__()
        self.media_path = media_path
        self.model_code = model_code
        self.is_running = True

    def stop(self):
        self.is_running = False

    def _tool_paths(self):
        ffmpeg = os.path.join(BASE_DIR, "tools", "ffmpeg", "ffmpeg.exe")
        whisper_cli = os.path.join(BASE_DIR, "tools", "whisper", "whisper-cli.exe")
        model_file = MODEL_FILE_MAP.get(self.model_code, "ggml-base.bin")
        model_path = os.path.join(BASE_DIR, "tools", "whisper", model_file)
        return ffmpeg, whisper_cli, model_path, model_file

    def run(self):
        try:
            ffmpeg, whisper_cli, model_path, model_file = self._tool_paths()

            if not os.path.exists(ffmpeg):
                raise Exception(
                    "缺少 ffmpeg.exe。\n"
                    f"请放到：{os.path.join(BASE_DIR, 'tools', 'ffmpeg')}\n"
                    "文件名必须是：ffmpeg.exe"
                )
            if not os.path.exists(whisper_cli):
                raise Exception(
                    "缺少 whisper-cli.exe。\n"
                    f"请放到：{os.path.join(BASE_DIR, 'tools', 'whisper')}\n"
                    "文件名必须是：whisper-cli.exe"
                )
            if not os.path.exists(model_path):
                raise Exception(
                    f"缺少模型文件：{model_file}\n"
                    f"请放到：{os.path.join(BASE_DIR, 'tools', 'whisper')}\n"
                    f"期望路径：{model_path}"
                )

            # -----------------------------
            # 1) 抽取音频
            # -----------------------------
            self.stage_signal.emit("抽取音频 {0}%")
            self.status_signal.emit("🎞️ 正在抽取音频...")
            self.progress_signal.emit(5)

            tmp_wav = os.path.join(tempfile.gettempdir(), f"love_{int(time.time())}.wav")
            cmd_ff = [
                ffmpeg, "-y",
                "-i", self.media_path,
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                "-f", "wav",
                tmp_wav
            ]

            print("[FFMPEG]", " ".join(cmd_ff))
            p = subprocess.run(
                cmd_ff,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace"
            )

            if p.returncode != 0 or (not os.path.exists(tmp_wav)):
                raise Exception("ffmpeg 抽音频失败：\n" + p.stdout[-2000:])

            if not self.is_running:
                return

            # -----------------------------
            # 2) whisper.cpp 转写
            # -----------------------------
            self.stage_signal.emit("识别中 {0}%")
            self.status_signal.emit("🧠 正在识别（离线）...")
            self.progress_signal.emit(15)

            out_dir = tempfile.gettempdir()
            out_prefix = os.path.join(out_dir, f"love_out_{int(time.time())}")
            out_txt = out_prefix + ".txt"

            cmd_wh = [
                whisper_cli,
                "-m", model_path,
                "-f", tmp_wav,
                "-l", "zh",
                "-otxt",
                "-of", out_prefix
            ]

            print("[WHISPER]", " ".join(cmd_wh))

            proc = subprocess.Popen(
                cmd_wh,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(whisper_cli),
                text=True,
                encoding="utf-8",
                errors="replace"
            )

            prog = 15
            last_lines = []
            while True:
                if not self.is_running:
                    try:
                        proc.kill()
                    except:
                        pass
                    return

                line = proc.stdout.readline()
                if not line:
                    break

                line = line.strip()
                if line:
                    last_lines.append(line)
                    if len(last_lines) > 60:
                        last_lines.pop(0)

                # 简单推进进度（不依赖特定输出格式，稳）
                if prog < 95:
                    prog += 1
                    self.progress_signal.emit(prog)

            code = proc.wait()
            if code != 0:
                raise Exception("whisper.cpp 识别失败：\n" + "\n".join(last_lines[-25:]))

            if not os.path.exists(out_txt):
                raise Exception(f"识别完成但未生成输出文件：{out_txt}")

            with open(out_txt, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()

            # 清理临时 wav（可选）
            try:
                os.remove(tmp_wav)
            except:
                pass

            self.progress_signal.emit(100)
            self.status_signal.emit("✅ 完成！")
            self.result_signal.emit(text)

        except Exception as e:
            print("[ERROR]", e)
            traceback.print_exc()
            self.error_signal.emit(str(e))

# ==============================================================================
# ✅ 主窗口
# ==============================================================================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("❤️ 专属助手（离线稳定版 whisper.cpp）")
        self.resize(1100, 700)
        self.setAcceptDrops(True)

        self.media_path = ""
        self.selected_model = "medium"
        self.worker = None
        self.model_btns = []

        self.init_ui()

    def init_ui(self):
        main = QHBoxLayout()
        left = QVBoxLayout()

        self.btn_import = QPushButton("\n📂 上传视频/音频\n(离线稳定版)\n")
        self.btn_import.setFixedHeight(140)
        self.btn_import.clicked.connect(self.sel_media)
        left.addWidget(self.btn_import)

        grid = QGridLayout()
        for i, m in enumerate(MODEL_OPTIONS):
            b = ModelCard(m["name"], m["desc"], m["code"], m["color"])
            b.clicked.connect(lambda c, x=b: self.on_clk(x))
            grid.addWidget(b, i // 2, i % 2)
            self.model_btns.append(b)
        left.addLayout(grid)
        self.on_clk(self.model_btns[0])

        self.lbl_stat = QLabel("准备就绪（请先放好 tools/ffmpeg 和 tools/whisper）")
        left.addWidget(self.lbl_stat)

        self.btn_start = ProgressButton("开始转换")
        self.btn_start.setFixedHeight(60)
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start)
        left.addWidget(self.btn_start)

        right = QVBoxLayout()
        self.txt = QTextEdit()
        right.addWidget(self.txt)

        btn_cp = QPushButton("📋 复制")
        btn_cp.clicked.connect(self.txt.selectAll)
        btn_cp.clicked.connect(self.txt.copy)
        right.addWidget(btn_cp)

        w_l = QWidget()
        w_l.setLayout(left)
        w_r = QWidget()
        w_r.setLayout(right)
        main.addWidget(w_l, 4)
        main.addWidget(w_r, 6)
        self.setLayout(main)

    def on_clk(self, b):
        for x in self.model_btns:
            x.setChecked(x == b)
            x.update_style(x == b)
        self.selected_model = b.code

    def dragEnterEvent(self, e):
        e.accept() if e.mimeData().hasUrls() else e.ignore()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            self.load(urls[0].toLocalFile())

    def sel_media(self):
        f, _ = QFileDialog.getOpenFileName(self, "选文件", "", "Media (*.mp4 *.mov *.avi *.mkv *.mp3 *.wav *.m4a)")
        if f:
            self.load(f)

    def load(self, p):
        self.media_path = p
        self.btn_import.setText(f"已加载: {os.path.basename(p)}")
        self.btn_start.setEnabled(True)

    def start(self):
        if not self.media_path:
            QMessageBox.warning(self, "提示", "请先选择文件")
            return

        self.btn_import.setEnabled(False)
        self.btn_start.start_processing()
        self.lbl_stat.setText("启动中...")

        self.worker = TranscribeThread(self.media_path, self.selected_model)
        self.worker.status_signal.connect(self.lbl_stat.setText)
        self.worker.progress_signal.connect(self.btn_start.set_progress)
        self.worker.stage_signal.connect(self.btn_start.set_format)
        self.worker.result_signal.connect(self.ok)
        self.worker.error_signal.connect(self.err)
        self.worker.start()

    def ok(self, t):
        self.btn_start.set_progress(100)
        self.txt.setPlainText(t)
        self.btn_import.setEnabled(True)
        self.btn_start.stop_processing()

    def err(self, m):
        self.btn_import.setEnabled(True)
        self.btn_start.stop_processing()
        self.lbl_stat.setText("❌ 出错")
        QMessageBox.warning(self, "错误", f"{m}\n\n看日志：{LOG_FILE}")

# ==============================================================================
# ✅ 程序入口
# ==============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
