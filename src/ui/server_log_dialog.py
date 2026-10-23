"""
後端伺服器日誌即時檢視器 (src/ui/server_log_dialog.py)
功能特點：
1. 即時讀取與呈現 logs/server.log (包含 uvicorn 請求、OpenVINO 模型載入、分層顯存配置與推論指標)
2. 語法色彩標註 (QSyntaxHighlighter 標註 INFO/WARNING/ERROR/關鍵字)
3. 關鍵字與日誌層級即時篩選
4. 自動捲動 (Auto-scroll)、即時動態追蹤 (Live Tail, 1.5s 輪詢)
5. 支援一鍵複製全部、開啟外部文字檔、清空日誌檔案
"""

import os
import re
import sys
from pathlib import Path
from typing import Optional, List

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPlainTextEdit,
    QLineEdit,
    QPushButton,
    QCheckBox,
    QComboBox,
    QLabel,
    QMessageBox,
    QApplication,
    QFrame,
)
from PyQt6.QtGui import (
    QSyntaxHighlighter,
    QTextCharFormat,
    QColor,
    QFont,
    QKeySequence,
    QShortcut,
)
from PyQt6.QtCore import Qt, QTimer, QRegularExpression


class ServerLogHighlighter(QSyntaxHighlighter):
    """日誌關鍵字語法高亮著色器"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlight_rules = []

        # 1. 時間戳記 (灰藍色)
        fmt_time = QTextCharFormat()
        fmt_time.setForeground(QColor("#64748B"))
        self.highlight_rules.append((r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d+)?", fmt_time))

        # 2. [INFO] (明亮天藍色)
        fmt_info = QTextCharFormat()
        fmt_info.setForeground(QColor("#38BDF8"))
        fmt_info.setFontWeight(QFont.Weight.Bold)
        self.highlight_rules.append((r"\[INFO\]", fmt_info))

        # 3. [WARNING] (警示橘黃色)
        fmt_warn = QTextCharFormat()
        fmt_warn.setForeground(QColor("#FBBF24"))
        fmt_warn.setFontWeight(QFont.Weight.Bold)
        self.highlight_rules.append((r"\[WARNING\]|WARNING|WARN", fmt_warn))

        # 4. [ERROR] / Exception / Traceback (危急紅色)
        fmt_err = QTextCharFormat()
        fmt_err.setForeground(QColor("#F87171"))
        fmt_err.setFontWeight(QFont.Weight.Bold)
        self.highlight_rules.append((r"\[ERROR\]|\[CRITICAL\]|ERROR|Exception|Traceback|Failed", fmt_err))

        # 5. [DEBUG] (淡灰色)
        fmt_debug = QTextCharFormat()
        fmt_debug.setForeground(QColor("#94A3B8"))
        self.highlight_rules.append((r"\[DEBUG\]", fmt_debug))

        # 6. HTTP 成功與端點 (綠色 / 紫色)
        fmt_http_ok = QTextCharFormat()
        fmt_http_ok.setForeground(QColor("#34D399"))
        self.highlight_rules.append((r'"HTTP/[^"]+" 200 OK', fmt_http_ok))

        fmt_keyword = QTextCharFormat()
        fmt_keyword.setForeground(QColor("#C084FC"))
        self.highlight_rules.append((r"\b(?:GPU|CPU|NPU|VRAM|OFFLOAD_RATIO|MoE|Qwen[^\s]+)\b", fmt_keyword))

    def highlightBlock(self, text: str):
        for pattern, fmt in self.highlight_rules:
            regex = QRegularExpression(pattern)
            match_iterator = regex.globalMatch(text)
            while match_iterator.hasNext():
                match = match_iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), fmt)


class ServerLogDialog(QDialog):
    """伺服器即時日誌檢視視窗"""

    def __init__(self, base_url: str = "http://127.0.0.1:12223", parent=None):
        super().__init__(parent)
        self.base_url = base_url
        self.setWindowTitle("📜 OpenVINO-Strata 後端伺服器日誌 (Server Log)")
        self.resize(980, 640)
        self.setMinimumSize(720, 450)

        # 尋找專案 logs/server.log
        root_dir = Path(__file__).resolve().parent.parent.parent
        self.log_file_path = root_dir / "logs" / "server.log"

        self.all_lines: List[str] = []
        self._last_file_mtime: float = 0.0

        self._setup_ui()
        self._setup_timer()
        self.refresh_logs(force_reload=True)

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 14, 16, 14)
        main_layout.setSpacing(10)

        # 1. 頂部工具列
        toolbar = QFrame()
        toolbar.setObjectName("LogToolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(8)

        # 關鍵字篩選輸入框
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 篩選關鍵字 (如 error, offload, GPU)...")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.textChanged.connect(self._apply_filter)
        toolbar_layout.addWidget(self.search_input, stretch=2)

        # 等級下拉選單
        self.level_combo = QComboBox()
        self.level_combo.addItems(["全部層級 (ALL)", "INFO", "WARNING", "ERROR"])
        self.level_combo.currentIndexChanged.connect(self._apply_filter)
        toolbar_layout.addWidget(self.level_combo)

        # 即時追蹤核取方塊
        self.live_tail_check = QCheckBox("即時監控 (Live)")
        self.live_tail_check.setChecked(True)
        self.live_tail_check.toggled.connect(self._on_live_toggled)
        toolbar_layout.addWidget(self.live_tail_check)

        # 自動捲動核取方塊
        self.auto_scroll_check = QCheckBox("自動置底")
        self.auto_scroll_check.setChecked(True)
        toolbar_layout.addWidget(self.auto_scroll_check)

        # 按鈕群
        self.refresh_btn = QPushButton("🔄 重新整理")
        self.refresh_btn.setObjectName("SecondaryBtn")
        self.refresh_btn.clicked.connect(lambda: self.refresh_logs(force_reload=True))
        toolbar_layout.addWidget(self.refresh_btn)

        self.copy_btn = QPushButton("📋 複製日誌")
        self.copy_btn.setObjectName("SecondaryBtn")
        self.copy_btn.clicked.connect(self._copy_logs)
        toolbar_layout.addWidget(self.copy_btn)

        self.open_file_btn = QPushButton("📁 開啟檔案")
        self.open_file_btn.setObjectName("SecondaryBtn")
        self.open_file_btn.setToolTip("以 Windows 預設文字編輯器開啟 server.log")
        self.open_file_btn.clicked.connect(self._open_log_file_externally)
        toolbar_layout.addWidget(self.open_file_btn)

        self.clear_btn = QPushButton("🗑 清空")
        self.clear_btn.setObjectName("SecondaryBtn")
        self.clear_btn.clicked.connect(self._clear_log_file)
        toolbar_layout.addWidget(self.clear_btn)

        main_layout.addWidget(toolbar)

        # 2. 中央日誌文字顯示區
        self.log_text = QPlainTextEdit(self)
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        font = QFont("Consolas", 11)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.log_text.setFont(font)
        self.log_text.setStyleSheet(
            "background-color: #0F1117; color: #E2E8F0; border: 1px solid #282D37; "
            "border-radius: 8px; padding: 10px; font-family: 'Consolas', 'Cascadia Code', monospace;"
        )

        # 附加語法著色器
        self.highlighter = ServerLogHighlighter(self.log_text.document())
        main_layout.addWidget(self.log_text, stretch=1)

        # 3. 底部狀態列
        status_layout = QHBoxLayout()
        self.status_label = QLabel("正在讀取日誌...")
        self.status_label.setStyleSheet("color: #94A3B8; font-size: 12px;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch(1)

        esc_shortcut = QShortcut(QKeySequence("Esc"), self)
        esc_shortcut.activated.connect(self.close)

        close_btn = QPushButton("關閉 (Esc)")
        close_btn.setObjectName("SecondaryBtn")
        close_btn.clicked.connect(self.close)
        status_layout.addWidget(close_btn)

        main_layout.addLayout(status_layout)

    def _setup_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer_tick)
        self.timer.start(1500)

    def _on_timer_tick(self):
        if self.live_tail_check.isChecked() and self.isVisible():
            self.refresh_logs(force_reload=False)

    def _on_live_toggled(self, checked: bool):
        if checked:
            self.status_label.setText("● 即時監控中 (每 1.5 秒更新)")
        else:
            self.status_label.setText("⏸ 已暫停即時監控")

    def refresh_logs(self, force_reload: bool = False):
        """讀取伺服器日誌檔案並更新畫面"""
        if not self.log_file_path.exists():
            self.log_text.setPlainText(
                f"[提示] 日誌檔案尚未產生 ({self.log_file_path})。\n"
                f"當後端伺服器 (run_server.py) 啟動、載入模型或執行推論時，將在此即時顯示完整運行資訊。\n\n"
                f"若伺服器未運行，請於終端執行: python run_tray.py 或 start_tray.bat"
            )
            self.status_label.setText("檔案狀態: 尚未產生日誌檔案")
            return

        try:
            stat = self.log_file_path.stat()
            mtime = stat.st_mtime

            # 若檔案無變更且非強制更新，略過讀取
            if not force_reload and mtime == self._last_file_mtime:
                return

            self._last_file_mtime = mtime

            with open(self.log_file_path, "r", encoding="utf-8", errors="replace") as f:
                # 保留最新 2500 行，避免介面渲染過載
                lines = f.readlines()
                self.all_lines = lines[-2500:] if len(lines) > 2500 else lines

            self._apply_filter()

            size_kb = stat.st_size / 1024.0
            self.status_label.setText(
                f"日誌路徑: {self.log_file_path.name} | 檔案大小: {size_kb:.1f} KB | "
                f"總行數: {len(self.all_lines)} 行 | 監控狀態: {'● Live' if self.live_tail_check.isChecked() else '⏸ 暫停'}"
            )

        except Exception as e:
            self.status_label.setText(f"讀取日誌失敗: {e}")

    def _apply_filter(self):
        """依據關鍵字與等級過濾日誌並呈現"""
        keyword = self.search_input.text().strip().lower()
        level_sel = self.level_combo.currentText()

        filtered = []
        for line in self.all_lines:
            # 等級篩選
            if level_sel == "INFO" and "[INFO]" not in line:
                continue
            elif level_sel == "WARNING" and "[WARNING]" not in line and "WARN" not in line:
                continue
            elif level_sel == "ERROR" and "[ERROR]" not in line and "Exception" not in line and "Traceback" not in line:
                continue

            # 關鍵字篩選
            if keyword and keyword not in line.lower():
                continue

            filtered.append(line)

        content = "".join(filtered)
        self.log_text.setPlainText(content)

        if self.auto_scroll_check.isChecked():
            cursor = self.log_text.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.log_text.setTextCursor(cursor)

    def _copy_logs(self):
        """複製當前檢視區日誌到剪貼簿"""
        text = self.log_text.toPlainText()
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(text)
            QMessageBox.information(self, "複製完成", "已將伺服器日誌複製至剪貼簿！")

    def _open_log_file_externally(self):
        """以系統預設應用程式開啟實體日誌檔案"""
        if not self.log_file_path.exists():
            QMessageBox.warning(self, "檔案不存在", f"日誌檔案尚未建立: {self.log_file_path}")
            return

        try:
            if sys.platform == "win32":
                os.startfile(str(self.log_file_path))
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(self.log_file_path)])
        except Exception as e:
            QMessageBox.warning(self, "無法開啟", f"開啟外部日誌檔案失敗: {e}")

    def _clear_log_file(self):
        """清空日誌檔案內容"""
        reply = QMessageBox.question(
            self,
            "確認清空",
            "確定要清空目前伺服器日誌檔案嗎？此操作無法復原。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                with open(self.log_file_path, "w", encoding="utf-8") as f:
                    f.truncate(0)
                self.all_lines.clear()
                self.log_text.clear()
                self._last_file_mtime = 0.0
                self.status_label.setText("日誌檔案已清空。")
            except Exception as e:
                QMessageBox.warning(self, "清空失敗", f"無法清空日誌檔案: {e}")

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_logs(force_reload=True)
        if hasattr(self, "timer") and not self.timer.isActive():
            self.timer.start(1500)

    def hideEvent(self, event):
        super().hideEvent(event)
        if hasattr(self, "timer") and self.timer.isActive():
            self.timer.stop()
