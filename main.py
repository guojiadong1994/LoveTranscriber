import sys
import os
import platform
import shutil
import traceback
import time
import subprocess
import tempfile
import json

# ==============================================================================
# 🛡️ 0. Ultra 9 / Windows 原生库防爆补丁（必须在任何大库 import 之前）
# ==============================================================================
def apply_ultra9_env_patch():
    # OpenMP / Intel OMP：禁用大小核绑核 + 降低抢占/等待问题
    os.environ["KMP_AFFINITY"] = "disabled"
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["KMP_BLOCKTIME"] = "0"
    os.environ["OMP_PROC_BIND"] = "FALSE"
    os.environ["OMP_PLACES"] = "cores"
    os.environ["OMP_DYNAMIC"] = "FALSE"

    # 线程数：初始化阶段强制单线程，避免 OMP 初始化阶段在大小核上抽风
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # 指令集降级（双保险）
    os.environ["MKL_ENABLE_INSTRUCTIONS"] = "AVX2"

    # HF 镜像与超时
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

apply_ultra9_env_patch()

# ==============================================================================
# 🛡️ 1. 日志与目录配置
# ==============================================================================
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_FILE = os.path.join(BASE_DIR, "crash.log")
MODELS_ROOT = os.path.join(BASE_DIR, "models")

import faulthandler
try:
    log_fs = open(LOG_FILE, "w", encoding="utf-8", buffering=1)
    sys.stdout = log_fs
    sys.stderr = log_fs
    faulthandler.enable(file=log_fs, all_threads=True)
    print(f"===== START {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
    print("Fix: Ultra9 env patch + worker subprocess isolation")
except:
    pass


# ==============================================================================
# ✅ GUI 依赖（仅 GUI 模式需要）
# ==============================================================================
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTextEdit, QMessageBox, QFileDialog, QGridLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPainter, QPainterPath
from PyQt6.QtCore import QRectF

# === 全局配置 ===
IS_MAC = (platform.system() == 'Darwin')
UI_FONT = "Microsoft YaHei" if not IS_MAC else "PingFang SC"

MODEL_MAP = {
    "medium": "systran/faster-whisper-medium",
    "base": "systran/faster-whisper-base",
    "large-v3": "systran/faster-whisper-large-v3",
    "small": "systran/faster-whisper-small"
}
MODEL_EXPECTED_SIZE = {"medium": 1500, "base": 145, "large-v3": 3050, "small": 480}
MODEL_OPTIONS = [
    {"name": "🌟 推荐模式", "desc": "精准与速度平衡", "code": "medium", "color": "#2ecc71"},
    {"name": "🚀 极速模式", "desc": "速度最快", "code": "base", "color": "#3498db"},
    {"name": "🧠 深度模式", "desc": "超准 but 稍慢", "code": "large-v3", "color": "#00cec9"},
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
# ✅ Worker 子进程线程：再怎么 access violation 也只崩子进程
# ==============================================================================
class WorkerProcessThread(QThread):
    status_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    stage_signal = pyqtSignal(str)
    result_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    dl_signal = pyqtSignal(int, int)  # downloaded_mb, expected_mb

    def __init__(self, video_path, model_code):
        super().__init__()
        self.video_path = video_path
        self.model_code = model_code
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            expected_mb = MODEL_EXPECTED_SIZE.get(self.model_code, 1000)
            self.status_signal.emit("⏳ 正在准备子进程...")
            self.progress_signal.emit(1)

            # 结果文件
            out_txt = os.path.join(tempfile.gettempdir(), f"love_transcribe_{int(time.time())}.txt")

            # 启动子进程（同一个 exe / 同一个 python）
            exe = sys.executable
            if getattr(sys, "frozen", False):
                args = [exe, "--worker", self.video_path, self.model_code, out_txt]
            else:
                script = os.path.abspath(__file__)
                args = [exe, script, "--worker", self.video_path, self.model_code, out_txt]

            env = os.environ.copy()
            # 再打一遍补丁，确保子进程一定吃到
            env["KMP_AFFINITY"] = "disabled"
            env["OMP_WAIT_POLICY"] = "PASSIVE"
            env["KMP_BLOCKTIME"] = "0"
            env["OMP_PROC_BIND"] = "FALSE"
            env["OMP_PLACES"] = "cores"
            env["OMP_DYNAMIC"] = "FALSE"
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["MKL_ENABLE_INSTRUCTIONS"] = "AVX2"
            env["HF_ENDPOINT"] = "https://hf-mirror.com"
            env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            env["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

            p = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=BASE_DIR,
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace"
            )

            # 解析子进程输出（JSON 行）
            self.stage_signal.emit("运行中 {0}%")
            self.progress_signal.emit(5)

            while True:
                if not self.is_running:
                    try:
                        p.kill()
                    except:
                        pass
                    return

                line = p.stdout.readline()
                if not line:
                    break
                line = line.strip()

                # 同步写入 crash.log
                try:
                    print("[WORKER]", line)
                except:
                    pass

                # 解析 JSON 行
                if line.startswith("{") and line.endswith("}"):
                    try:
                        msg = json.loads(line)
                    except:
                        continue

                    t = msg.get("type")
                    if t == "status":
                        self.status_signal.emit(msg.get("text", ""))
                    elif t == "progress":
                        self.progress_signal.emit(int(msg.get("value", 0)))
                    elif t == "download":
                        self.dl_signal.emit(int(msg.get("mb", 0)), int(msg.get("expected", expected_mb)))
                    elif t == "stage":
                        self.stage_signal.emit(msg.get("fmt", "运行中 {0}%"))
                    elif t == "error":
                        self.error_signal.emit(msg.get("text", "未知错误"))
                    continue

            code = p.wait()

            # 0xC0000005 access violation 通常会是非 0 退出码（有时是 -1073741819）
            if code != 0:
                self.error_signal.emit(
                    f"子进程异常退出 (exit={code})：\n"
                    f"这通常是原生库 Access Violation。\n"
                    f"请把 {LOG_FILE} 发我。"
                )
                return

            if not os.path.exists(out_txt):
                self.error_signal.emit("子进程未生成结果文件，可能中途崩溃。请看 crash.log")
                return

            with open(out_txt, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            self.progress_signal.emit(100)
            self.status_signal.emit("✅ 完成！")
            self.result_signal.emit(text)

        except Exception as e:
            self.error_signal.emit(f"主进程异常：{e}\n看日志 crash.log")


# ==============================================================================
# ✅ 主窗口
# ==============================================================================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("❤️ 专属助手 (Ultra9 最终修复-子进程隔离版)")
        self.resize(1100, 700)
        self.setAcceptDrops(True)
        self.video_path = ""
        self.selected_model = "medium"
        self.worker = None
        self.model_btns = []
        self.init_ui()

    def init_ui(self):
        main = QHBoxLayout()
        left = QVBoxLayout()

        self.btn_import = QPushButton("\n📂 上传视频\n(子进程防崩版)\n")
        self.btn_import.setFixedHeight(140)
        self.btn_import.clicked.connect(self.sel_video)
        left.addWidget(self.btn_import)

        grid = QGridLayout()
        for i, m in enumerate(MODEL_OPTIONS):
            b = ModelCard(m["name"], m["desc"], m["code"], m["color"])
            b.clicked.connect(lambda c, x=b: self.on_clk(x))
            grid.addWidget(b, i // 2, i % 2)
            self.model_btns.append(b)
        left.addLayout(grid)
        self.on_clk(self.model_btns[0])

        self.lbl_stat = QLabel("准备就绪")
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
        self.load(e.mimeData().urls()[0].toLocalFile())

    def sel_video(self):
        f, _ = QFileDialog.getOpenFileName(self, "选文件", "", "Media (*.mp4 *.mov *.avi *.mp3)")
        if f:
            self.load(f)

    def load(self, p):
        self.video_path = p
        self.btn_import.setText(f"已加载: {os.path.basename(p)}")
        self.btn_start.setEnabled(True)

    def start(self):
        if not self.video_path:
            QMessageBox.warning(self, "提示", "请先选择文件")
            return

        self.btn_import.setEnabled(False)
        self.btn_start.start_processing()

        self.worker = WorkerProcessThread(self.video_path, self.selected_model)
        self.worker.status_signal.connect(self.lbl_stat.setText)
        self.worker.progress_signal.connect(self.btn_start.set_progress)
        self.worker.stage_signal.connect(self.btn_start.set_format)
        self.worker.result_signal.connect(self.ok)
        self.worker.error_signal.connect(self.err)
        self.worker.dl_signal.connect(self.on_dl)
        self.worker.start()

    def on_dl(self, mb, expected):
        # 显示“下载 xxM/yyM”
        self.btn_start.set_text_override(f"下载 {mb}M/{expected}M")

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
# ✅ 子进程 worker 入口：这里允许崩（崩了也不带走 GUI）
# ==============================================================================
def worker_main(video_path, model_code, out_txt):
    # 注意：worker 模式下才 import 这些重库，避免污染 GUI 进程
    apply_ultra9_env_patch()

    repo_id = MODEL_MAP[model_code]
    models_root = os.path.join(BASE_DIR, "models")
    os.makedirs(models_root, exist_ok=True)
    model_base_dir = os.path.join(models_root, f"models--{repo_id.replace('/', '--')}")
    expected_mb = MODEL_EXPECTED_SIZE.get(model_code, 1000)

    def jprint(obj):
        # worker -> GUI：统一 JSON 行
        print(json.dumps(obj, ensure_ascii=False), flush=True)

    jprint({"type": "status", "text": "⏳ 正在校验/下载模型..."})
    jprint({"type": "download", "mb": 0, "expected": expected_mb})
    jprint({"type": "progress", "value": 5})
    jprint({"type": "stage", "fmt": "运行中 {0}%"})

    # 延迟 import
    from huggingface_hub import snapshot_download
    from faster_whisper import WhisperModel

    # 下载（huggingface 会自动 resume）
    real_model_path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=model_base_dir,
        max_workers=1
    )

    jprint({"type": "status", "text": "🧠 正在唤醒 AI 引擎..."})
    jprint({"type": "progress", "value": 40})

    # 关键：依次尝试不同 compute_type（某些机器 int8 内核更容易炸）
    compute_try = ["int8", "int8_float32", "float32"]
    last_err = None
    model = None

    for ct in compute_try:
        try:
            jprint({"type": "status", "text": f"🔧 加载模型 compute_type={ct} ..."})
            model = WhisperModel(
                real_model_path,
                device="cpu",
                compute_type=ct,
                cpu_threads=1,
                local_files_only=True
            )
            break
        except Exception as e:
            last_err = e
            jprint({"type": "status", "text": f"⚠️ 加载失败，尝试降级：{ct} -> next"})
            continue

    if model is None:
        jprint({"type": "error", "text": f"模型加载失败：{last_err}"})
        sys.exit(2)

    jprint({"type": "status", "text": "🎧 正在分析..."})
    jprint({"type": "progress", "value": 55})

    # transcribe 建议也尽量减少并发
    segments, info = model.transcribe(
        video_path,
        beam_size=5,
        language="zh",
        initial_prompt="这是一段清晰的普通话，请加标点符号。",
        vad_filter=False,
        condition_on_previous_text=True
    )

    full_text = ""
    dur = float(getattr(info, "duration", 0.0) or 0.0)

    for seg in segments:
        full_text += seg.text
        if dur > 0:
            pct = 55 + int((float(seg.end) / dur) * 44)
            jprint({"type": "progress", "value": min(99, pct)})

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(full_text)

    jprint({"type": "progress", "value": 100})
    jprint({"type": "status", "text": "✅ 完成！"})
    sys.exit(0)


# ==============================================================================
# ✅ 程序入口
# ==============================================================================
if __name__ == "__main__":
    # 子进程 worker 模式
    if len(sys.argv) >= 5 and sys.argv[1] == "--worker":
        _, _, vpath, mcode, outtxt = sys.argv[:5]
        worker_main(vpath, mcode, outtxt)

    # GUI 模式
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
