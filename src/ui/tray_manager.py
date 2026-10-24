"""
Windows 系統匣管理員 (src/ui/tray_manager.py)
負責：
1. Windows 系統匣常駐圖示 (QSystemTrayIcon) 與狀態提示氣泡
2. 動態即時狀態顯示 (後端連線、當前載入模型、GPU VRAM 顯存、Offload 比例)
3. 系統匣右鍵選單 (開啟/隱藏 Chat UI、Web API 文檔、一鍵熱切換模型、一鍵釋放顯存、偏好設定、結束程式)
4. 單擊/雙擊系統匣圖示秒開/喚醒 Chat UI
5. 視窗關閉時自動縮小至系統匣背景運作
"""

import logging
import os
import sys
import webbrowser
from pathlib import Path
from typing import Optional, Dict, Any, List

from PyQt6.QtWidgets import (
    QApplication,
    QSystemTrayIcon,
    QMenu,
    QMessageBox,
)
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont, QAction, QActionGroup
from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal

from src.ui.api_worker import HealthCheckWorker, ModelControlWorker
from src.utils.paths import get_asset_path

logger = logging.getLogger("ov_strata.tray")


def get_app_icon() -> QIcon:
    """取得專案高解析度圖示，若檔案不存在則自動以 QPainter 程式化生成科技感備援圖示"""
    ico_path = get_asset_path("icon.ico")
    png_path = get_asset_path("icon.png")

    if ico_path.exists():
        icon = QIcon(str(ico_path))
        if not icon.isNull():
            return icon

    if png_path.exists():
        icon = QIcon(str(png_path))
        if not icon.isNull():
            return icon

    # 備援：動態以 QPainter 繪製科技藍六角微光晶片圖示
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # 底板圓角
    painter.setBrush(QColor(15, 23, 42))
    painter.setPen(QColor(56, 189, 248))
    painter.drawRoundedRect(4, 4, 56, 56, 12, 12)

    # 科技藍與青色分層
    painter.setBrush(QColor(14, 165, 233))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(14, 18, 36, 6, 3, 3)

    painter.setBrush(QColor(59, 130, 246))
    painter.drawRoundedRect(11, 28, 42, 6, 3, 3)

    painter.setBrush(QColor(139, 92, 246))
    painter.drawRoundedRect(16, 38, 32, 6, 3, 3)

    painter.end()
    return QIcon(pixmap)


class StrataTrayManager(QObject):
    """OpenVINO-Strata 專用 Windows 系統匣常駐管理員"""

    status_changed = pyqtSignal(bool, dict)

    def __init__(
        self,
        app: QApplication,
        window=None,
        base_url: str = "http://127.0.0.1:12223",
        server_process=None,
        minimize_on_close: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.app = app
        self.window = window
        self.base_url = base_url.rstrip("/")
        self.server_process = server_process
        self.minimize_on_close = minimize_on_close
        self.has_shown_minimize_tip = False

        self.is_server_online = False
        self.current_model: Optional[str] = None
        self.available_models: List[Dict[str, Any]] = []
        self.server_info: Dict[str, Any] = {}

        self.health_worker: Optional[HealthCheckWorker] = None
        self.control_worker: Optional[ModelControlWorker] = None

        self.icon = get_app_icon()

        if window:
            window.setWindowIcon(self.icon)

        self._setup_tray()
        # 若無外部 window 負責輪詢，才由 tray_manager 獨立運行定時檢查
        if not self.window:
            self._setup_health_timer()

    def _setup_tray(self):
        if not QSystemTrayIcon.isSystemTrayAvailable():
            logger.warning("目前系統不支援系統匣 (System Tray)，略過系統匣初始化。")
            return

        self.tray_icon = QSystemTrayIcon(self.icon, self.app)
        self.tray_icon.setToolTip("OpenVINO-Strata AI Studio\n伺服器連線檢查中...")

        # 建立快顯選單
        self.menu = QMenu()
        self._build_menu()

        self.tray_icon.setContextMenu(self.menu)
        self.tray_icon.activated.connect(self._on_tray_activated)
        self.tray_icon.show()

    def _build_menu(self):
        self.menu.clear()

        # 1. 主視窗控制操作
        self.action_open_ui = QAction("💬 開啟 Chat UI", self.menu)
        font = self.action_open_ui.font()
        font.setBold(True)
        self.action_open_ui.setFont(font)
        self.action_open_ui.triggered.connect(self.show_chat_ui)
        self.menu.addAction(self.action_open_ui)

        self.action_hide_ui = QAction("🙈 隱藏 Chat UI", self.menu)
        self.action_hide_ui.triggered.connect(self.hide_chat_ui)
        self.menu.addAction(self.action_hide_ui)

        self.action_open_docs = QAction("🌐 開啟 Web API 文檔 (Swagger)", self.menu)
        self.action_open_docs.triggered.connect(self._on_open_docs)
        self.menu.addAction(self.action_open_docs)

        self.menu.addSeparator()

        # 2. 即時狀態資訊 (不可點選之資訊指示項)
        self.action_status_info = QAction("⚡ 伺服器狀態: 連線檢查中...", self.menu)
        self.action_status_info.setEnabled(False)
        self.menu.addAction(self.action_status_info)

        self.action_model_info = QAction("🧠 當前模型: (未載入)", self.menu)
        self.action_model_info.setEnabled(False)
        self.menu.addAction(self.action_model_info)

        self.action_mem_info = QAction("📊 顯存分層: VRAM: -- MB | Offload: --%", self.menu)
        self.action_mem_info.setEnabled(False)
        self.menu.addAction(self.action_mem_info)

        self.menu.addSeparator()

        # 3. 模型熱切換子選單 (動態填入)
        self.model_submenu = self.menu.addMenu("🔄 切換模型 (Switch Model)")
        self._refresh_model_submenu()

        # 4. 顯存釋放與設定
        self.action_unload = QAction("🧹 釋放顯存 (Unload VRAM)", self.menu)
        self.action_unload.setToolTip("卸載目前模型並立即釋放 GPU 顯存與系統 RAM")
        self.action_unload.triggered.connect(self.unload_vram)
        self.menu.addAction(self.action_unload)

        self.action_logs = QAction("📜 檢視伺服器日誌 (Server Log)...", self.menu)
        self.action_logs.triggered.connect(self._on_open_logs)
        self.menu.addAction(self.action_logs)

        self.action_settings = QAction("⚙️ 偏好設定 (Settings)...", self.menu)
        self.action_settings.triggered.connect(self._on_open_settings)
        self.menu.addAction(self.action_settings)

        self.menu.addSeparator()

        # 5. 行為選項
        self.action_minimize_toggle = QAction("☑️ 關閉視窗時最小化至系統匣", self.menu)
        self.action_minimize_toggle.setCheckable(True)
        self.action_minimize_toggle.setChecked(self.minimize_on_close)
        self.action_minimize_toggle.triggered.connect(self._on_toggle_minimize_on_close)
        self.menu.addAction(self.action_minimize_toggle)

        from src.utils.autostart import is_autostart_supported, is_autostart_enabled
        self.action_autostart_toggle = QAction("☑️ 登入 Windows 時自動在背景啟動", self.menu)
        self.action_autostart_toggle.setCheckable(True)
        self.action_autostart_toggle.setChecked(is_autostart_enabled())
        self.action_autostart_toggle.setEnabled(is_autostart_supported())
        self.action_autostart_toggle.triggered.connect(self._on_toggle_autostart)
        self.menu.addAction(self.action_autostart_toggle)

        self.menu.addSeparator()

        # 6. 結束程式
        self.action_exit = QAction("🚪 結束 OpenVINO-Strata (Exit)", self.menu)
        self.action_exit.triggered.connect(self.quit_application)
        self.menu.addAction(self.action_exit)

    def action_minimize_on_close_action(self) -> QAction:
        return self.action_minimize_toggle

    def _setup_health_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_health)
        self.timer.start(4000)
        self.check_health()

    def check_health(self):
        if self.health_worker and self.health_worker.isRunning():
            return
        self.health_worker = HealthCheckWorker(self.base_url)
        self.health_worker.status_updated.connect(self._on_health_updated)
        self.health_worker.start()

    def _on_health_updated(self, is_online: bool, data: dict):
        self.is_server_online = is_online
        self.server_info = data
        self.status_changed.emit(is_online, data)

        if is_online:
            device = data.get("device", "GPU")
            model = data.get("model") or "無"
            loaded = data.get("loaded", False)
            offload = data.get("offload_ratio", 0)
            gpu_mem = data.get("gpu_memory", {})
            sys_mem = data.get("system_memory", {})
            vram_mb = gpu_mem.get("allocated_memory_mb", 0.0)
            ram_avail = sys_mem.get("available_ram_gb", 0.0)

            self.current_model = model if loaded else None
            self.available_models = data.get("available_models", [])

            self.action_status_info.setText(f"⚡ 伺服器狀態: ● 已連線 ({device})")
            if loaded:
                self.action_model_info.setText(f"🧠 當前模型: {model}")
                self.action_mem_info.setText(
                    f"📊 顯存: {vram_mb:.0f} MB | Offload: {offload}% | 可用 RAM: {ram_avail:.1f} GB"
                )
                tooltip = (
                    f"OpenVINO-Strata AI Studio\n"
                    f"---------------------------\n"
                    f"狀態: ● 已連線 ({device})\n"
                    f"模型: {model}\n"
                    f"顯存: {vram_mb:.0f} MB (Offload: {offload}%)\n"
                    f"可用 RAM: {ram_avail:.1f} GB\n"
                    f"(按一下開啟 Chat UI)"
                )
            else:
                self.action_model_info.setText("🧠 當前模型: (已卸載/待命)")
                self.action_mem_info.setText(
                    f"📊 顯存: 0 MB (未載入) | 可用 RAM: {ram_avail:.1f} GB"
                )
                tooltip = (
                    f"OpenVINO-Strata AI Studio\n"
                    f"---------------------------\n"
                    f"狀態: ● 已連線 ({device} 待命中)\n"
                    f"可用 RAM: {ram_avail:.1f} GB\n"
                    f"(按一下開啟 Chat UI)"
                )
        else:
            self.current_model = None
            self.available_models = []
            self.action_status_info.setText("⚡ 伺服器狀態: ● 未連線 (Offline)")
            self.action_model_info.setText("🧠 當前模型: --")
            self.action_mem_info.setText("📊 顯存分層: --")
            tooltip = (
                f"OpenVINO-Strata AI Studio\n"
                f"---------------------------\n"
                f"狀態: ● 未連線 ({self.base_url})\n"
                f"請確認後端服務已啟動"
            )

        if hasattr(self, "tray_icon"):
            self.tray_icon.setToolTip(tooltip)

        self._refresh_model_submenu()

    def _refresh_model_submenu(self):
        if not hasattr(self, "model_submenu"):
            return

        self.model_submenu.clear()

        if not self.is_server_online:
            act = self.model_submenu.addAction("(伺服器離線中)")
            act.setEnabled(False)
            return

        if not self.available_models:
            act = self.model_submenu.addAction("(未搜尋到可用的 OpenVINO 模型)")
            act.setEnabled(False)
            return

        for m in self.available_models:
            m_id = m.get("id", "")
            is_cur = (m_id == self.current_model)
            prefix = "✔ " if is_cur else "   "
            rec_off = m.get("recommended_offload_ratio", 0)
            text = f"{prefix}{m_id} (Offload ~{rec_off}%)"

            action = QAction(text, self.model_submenu)
            action.setData(m_id)
            if is_cur:
                f = action.font()
                f.setBold(True)
                action.setFont(f)

            # 使用 lambda 閉包綁定 model_id
            action.triggered.connect(lambda checked=False, target=m_id: self.switch_model(target))
            self.model_submenu.addAction(action)

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason):
        """處理單擊或雙擊系統匣圖示動作：智慧切換顯示/隱藏"""
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self.toggle_chat_ui()

    def toggle_chat_ui(self):
        """切換 Chat UI 顯示狀態"""
        if not self.window:
            return

        if self.window.isVisible() and not self.window.isMinimized():
            self.hide_chat_ui()
        else:
            self.show_chat_ui()

    def show_chat_ui(self):
        """開啟並帶至前景顯示 Chat UI"""
        if not self.window:
            return

        # 若處於最小化狀態，還原視窗
        if self.window.isMinimized():
            self.window.setWindowState(
                self.window.windowState() & ~Qt.WindowState.WindowMinimized | Qt.WindowState.WindowActive
            )
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()

        # 焦點自動帶入輸入框
        if hasattr(self.window, "input_box"):
            self.window.input_box.setFocus()

    def hide_chat_ui(self):
        """隱藏 Chat UI 至系統匣"""
        if self.window and self.window.isVisible():
            self.window.hide()

    def _on_open_docs(self):
        """開啟瀏覽器前往 Swagger API 文件"""
        docs_url = f"{self.base_url}/docs"
        webbrowser.open(docs_url)

    def _on_open_settings(self):
        """開啟設定對話框"""
        if self.window and hasattr(self.window, "_on_open_settings"):
            self.show_chat_ui()
            self.window._on_open_settings()
        elif hasattr(self, "show_notification"):
            self.show_notification("OpenVINO-Strata", "請先開啟 Chat UI 進行設定。")

    def _on_open_logs(self):
        """開啟伺服器日誌檢視器"""
        if self.window and hasattr(self.window, "_on_open_server_log"):
            self.window._on_open_server_log()
        else:
            from src.ui.server_log_dialog import ServerLogDialog
            if not hasattr(self, "_log_dialog") or self._log_dialog is None:
                self._log_dialog = ServerLogDialog(base_url=self.base_url)
            self._log_dialog.show()
            self._log_dialog.raise_()
            self._log_dialog.activateWindow()

    def _on_toggle_minimize_on_close(self, checked: bool):
        self.minimize_on_close = checked
        if self.window and hasattr(self.window, "minimize_to_tray_on_close"):
            self.window.minimize_to_tray_on_close = checked

    def _on_toggle_autostart(self, checked: bool):
        """切換 Windows 使用者登入後自動啟動"""
        from src.utils.autostart import set_autostart
        ok, msg = set_autostart(checked)
        if ok:
            status_text = "已開啟" if checked else "已關閉"
            self.show_notification("Windows 登入自啟動", f"{status_text} Windows 登入自動在背景啟動。")
        else:
            self.action_autostart_toggle.setChecked(not checked)
            self.show_notification("自啟動設定失敗", msg, QSystemTrayIcon.MessageIcon.Warning)

    def switch_model(self, model_id: str):
        """從系統匣一鍵切換模型"""
        if not model_id:
            return
        if self.control_worker and self.control_worker.isRunning():
            self.show_notification("OpenVINO-Strata", "正在處理其他模型控制操作，請稍候...")
            return

        self.show_notification("OpenVINO-Strata", f"⏳ 正在載入模型 [{model_id}]...")

        device = "GPU"
        vram_budget = None
        offload_ratio = -1
        kv_cache = "u8"
        enable_mtp = True

        if self.window and hasattr(self.window, "config"):
            device = self.window.config.get("device", device)
            vram_budget = self.window.config.get("vram_budget_gb")
            offload_ratio = self.window.config.get("offload_ratio", offload_ratio)
            kv_cache = self.window.config.get("kv_cache_precision", kv_cache)
            enable_mtp = self.window.config.get("enable_mtp", enable_mtp)

        payload = {
            "model": model_id,
            "device": device,
            "vram_budget_gb": vram_budget,
            "offload_ratio": offload_ratio,
            "kv_cache_precision": kv_cache,
            "enable_mtp": enable_mtp,
        }

        self.control_worker = ModelControlWorker(self.base_url, "load", payload)
        self.control_worker.completed.connect(self._on_model_switch_completed)
        self.control_worker.start()

    def _on_model_switch_completed(self, ok: bool, msg: str, data: dict):
        if ok:
            model = data.get("model", "")
            self.show_notification("OpenVINO-Strata", f"🎉 模型已就緒：{model}\n{msg}")
        else:
            self.show_notification(
                "OpenVINO-Strata", f"❌ 載入模型失敗:\n{msg}", QSystemTrayIcon.MessageIcon.Warning
            )
        self.check_health()

    def unload_vram(self):
        """從系統匣一鍵釋放 GPU 顯存"""
        if self.control_worker and self.control_worker.isRunning():
            return

        self.control_worker = ModelControlWorker(self.base_url, "unload")
        self.control_worker.completed.connect(self._on_unload_completed)
        self.control_worker.start()

    def _on_unload_completed(self, ok: bool, msg: str, data: dict):
        if ok:
            self.show_notification("OpenVINO-Strata", "🧹 顯存已成功釋放！GPU 與 RAM 已歸還系統。")
        else:
            self.show_notification(
                "OpenVINO-Strata", f"釋放顯存失敗: {msg}", QSystemTrayIcon.MessageIcon.Warning
            )
        self.check_health()

    def show_notification(
        self,
        title: str,
        message: str,
        icon_type: QSystemTrayIcon.MessageIcon = QSystemTrayIcon.MessageIcon.Information,
        timeout_ms: int = 3500,
    ):
        """彈出 Windows 系統匣通知氣泡"""
        if hasattr(self, "tray_icon") and self.tray_icon.isVisible():
            self.tray_icon.showMessage(title, message, icon_type, timeout_ms)

    def on_window_minimized_to_tray(self):
        """當視窗關閉或縮小至系統匣時呼叫"""
        if not self.has_shown_minimize_tip:
            self.show_notification(
                "OpenVINO-Strata 正在背景執行",
                "應用程式已縮小至工作列右下角系統匣。\n隨時按一下系統匣圖示即可重新喚醒 Chat UI。",
            )
            self.has_shown_minimize_tip = True

    def quit_application(self):
        """完整結束 OpenVINO-Strata 應用程式"""
        # 關閉計時器
        if hasattr(self, "timer") and self.timer.isActive():
            self.timer.stop()

        # 安全解除背景工作執行緒訊號連接
        if self.health_worker:
            try:
                self.health_worker.status_updated.disconnect()
            except Exception:
                pass
        if self.control_worker:
            try:
                self.control_worker.completed.disconnect()
            except Exception:
                pass

        # 關閉視窗
        if self.window:
            self.window.minimize_to_tray_on_close = False
            if hasattr(self.window, "health_timer") and self.window.health_timer.isActive():
                self.window.health_timer.stop()
            if hasattr(self.window, "health_worker") and self.window.health_worker:
                try:
                    self.window.health_worker.status_updated.disconnect()
                except Exception:
                    pass
            self.window.close()

        # 隱藏系統匣圖示
        if hasattr(self, "tray_icon"):
            self.tray_icon.hide()

        # 若有託管的後端伺服器程序，一併終止
        if self.server_process:
            try:
                logger.info("正在停止背景後端伺服器 (PID: %s)...", self.server_process.pid)
                self.server_process.terminate()
                self.server_process.wait(timeout=3)
            except Exception as e:
                logger.warning("強制終止伺服器: %s", e)
                try:
                    self.server_process.kill()
                except Exception:
                    pass

        self.app.quit()
