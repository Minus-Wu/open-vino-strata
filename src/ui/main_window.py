"""
PyQt6 桌面聊天室主視窗 (src/ui/main_window.py)
整合：
1. 頂部 PolyStrata 模型動態選擇器、一鍵載入/切換、一鍵釋放顯存 (Unload VRAM) 與即時顯存監控膠囊
2. 中央對話捲動區 (支援 Markdown、<think> 深度思考摺疊面板與 <tool_call> 標籤)
3. 底部多模態輸入區 (支援附加圖片 Base64 Vision 輸入、單擊切換思考模式、Enter 發送與隨時中斷生成)
"""

import base64
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QLabel,
    QPushButton,
    QPlainTextEdit,
    QComboBox,
    QCheckBox,
    QFileDialog,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSettings, QRect
from PyQt6.QtGui import QKeyEvent, QGuiApplication

from src.ui.chat_widget import ChatHistoryWidget, MessageBubble
from src.ui.api_worker import ChatStreamWorker, HealthCheckWorker, ModelControlWorker
from src.ui.settings_dialog import SettingsDialog
from src.ui.server_log_dialog import ServerLogDialog
from src.ui.tray_manager import StrataTrayManager, get_app_icon


class ChatInputBox(QPlainTextEdit):
    """自訂輸入框：Enter 直接發送，Shift + Enter 換行"""

    return_pressed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ChatInput")
        self.setPlaceholderText(
            "輸入訊息... (Enter 發送，Shift + Enter 換行；可勾選『💭 思考』或『🖼️ 圖片』進行多模態推理)"
        )
        self.setFixedHeight(68)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                self.return_pressed.emit()
                event.accept()
                return
        super().keyPressEvent(event)


class MainWindow(QMainWindow):
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:12223",
        enable_tray: bool = True,
        minimize_on_close: bool = True,
    ):
        super().__init__()
        self.setWindowTitle("OpenVINO-Strata — Low-VRAM 35B-A3B MoE AI Studio")
        self.resize(1120, 780)
        self.setMinimumSize(820, 560)
        self.setWindowIcon(get_app_icon())

        self.minimize_to_tray_on_close = minimize_on_close
        self.tray_manager: Optional[StrataTrayManager] = None

        self.config: Dict[str, Any] = {
            "base_url": base_url,
            "system_prompt": "You are a helpful, accurate, and concise AI assistant.",
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "device": "GPU",
            "vram_budget_gb": None,
            "offload_ratio": -1,
            "kv_cache_precision": "u8",
            "enable_mtp": True,
            "minimize_to_tray": minimize_on_close,
        }

        self.messages: List[Dict[str, Any]] = []
        self.current_worker: Optional[ChatStreamWorker] = None
        self.current_assistant_bubble: Optional[MessageBubble] = None
        self.health_worker: Optional[HealthCheckWorker] = None
        self.control_worker: Optional[ModelControlWorker] = None
        self.attached_image_path: Optional[Path] = None

        self._setup_ui()
        self._setup_health_timer()
        self._restore_window_state()

        if enable_tray:
            app = QApplication.instance()
            if app:
                self.tray_manager = StrataTrayManager(
                    app=app,
                    window=self,
                    base_url=self.config["base_url"],
                    minimize_on_close=self.minimize_to_tray_on_close,
                    parent=self,
                )

    def _setup_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. 頂部標題與 Strata 狀態控制列
        header = QFrame()
        header.setObjectName("HeaderFrame")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 10, 16, 10)
        header_layout.setSpacing(8)

        title_label = QLabel("⚡ OpenVINO-Strata")
        title_label.setObjectName("TitleLabel")
        header_layout.addWidget(title_label)

        self.status_badge = QLabel("● 連線檢查中...")
        self.status_badge.setObjectName("StatusBadge")
        self.status_badge.setProperty("online", "false")
        header_layout.addWidget(self.status_badge)

        self.strata_badge = QLabel("Offload: --% | VRAM: -- MB")
        self.strata_badge.setObjectName("StrataBadge")
        header_layout.addWidget(self.strata_badge)

        header_layout.addStretch(1)

        # 模型選擇下拉選單與載入/卸載按鈕
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(230)
        self.model_combo.setToolTip("選擇要載入或熱切換的本地 OpenVINO 模型")
        header_layout.addWidget(self.model_combo)

        self.load_btn = QPushButton("🔄 載入")
        self.load_btn.setObjectName("SecondaryBtn")
        self.load_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.load_btn.setToolTip("以目前的 Strata 顯存分層設定載入/切換至所選模型")
        self.load_btn.clicked.connect(self._on_load_selected_model)
        header_layout.addWidget(self.load_btn)

        self.unload_btn = QPushButton("🧹 釋放顯存")
        self.unload_btn.setObjectName("UnloadBtn")
        self.unload_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.unload_btn.setToolTip("立即卸載目前模型並將 GPU VRAM 與系統 RAM 歸還作業系統")
        self.unload_btn.clicked.connect(self._on_unload_model)
        header_layout.addWidget(self.unload_btn)

        clear_btn = QPushButton("🗑 清除")
        clear_btn.setObjectName("SecondaryBtn")
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.clicked.connect(self._on_clear_chat)
        header_layout.addWidget(clear_btn)

        self.log_btn = QPushButton("📜 日誌")
        self.log_btn.setObjectName("SecondaryBtn")
        self.log_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.log_btn.setToolTip("檢視後端推論伺服器即時運行日誌 (Server Log)")
        self.log_btn.clicked.connect(self._on_open_server_log)
        header_layout.addWidget(self.log_btn)

        settings_btn = QPushButton("⚙️ 設定")
        settings_btn.setObjectName("SecondaryBtn")
        settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        settings_btn.clicked.connect(self._on_open_settings)
        header_layout.addWidget(settings_btn)

        main_layout.addWidget(header)

        # 2. 中央對話歷史捲動區
        self.chat_history = ChatHistoryWidget(self)
        main_layout.addWidget(self.chat_history, stretch=1)

        welcome = self.chat_history.add_message(
            "assistant",
            "👋 歡迎使用 **OpenVINO-Strata AI Studio**！\n\n"
            "本系統整合 **Intel OpenVINO 2026** 與 **PolyStrata 分層記憶體架構**：\n"
            "- 透過 `OFFLOAD_RATIO` 將 **Qwen3.8-35B-A3B / Qwen3.6-35B-A3B** 的 256 個專家層動態分層於 **GPU VRAM** 與 **System RAM**\n"
            "- 支援 `<think>` 深度思考鏈摺疊檢視、`<tool_call>` 工具呼叫解析、多模態圖片輸入與一鍵釋放顯存。",
        )
        welcome.metrics_label.hide()

        # 3. 底部輸入控制區
        bottom_wrapper = QWidget()
        bottom_layout = QVBoxLayout(bottom_wrapper)
        bottom_layout.setContentsMargins(18, 6, 18, 14)
        bottom_layout.setSpacing(6)

        # 圖片附件預覽列 (預設隱藏)
        self.attach_bar = QFrame()
        attach_bar_layout = QHBoxLayout(self.attach_bar)
        attach_bar_layout.setContentsMargins(8, 2, 8, 2)
        self.attach_label = QLabel("")
        self.attach_label.setStyleSheet("color: #60A5FA; font-size: 12px;")
        remove_attach_btn = QPushButton("✕ 移除圖片")
        remove_attach_btn.setObjectName("SecondaryBtn")
        remove_attach_btn.setFixedHeight(24)
        remove_attach_btn.clicked.connect(self._clear_attached_image)
        attach_bar_layout.addWidget(self.attach_label)
        attach_bar_layout.addStretch(1)
        attach_bar_layout.addWidget(remove_attach_btn)
        self.attach_bar.hide()
        bottom_layout.addWidget(self.attach_bar)

        input_frame = QFrame()
        input_frame.setObjectName("InputFrame")
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(8, 6, 8, 6)
        input_layout.setSpacing(10)

        self.input_box = ChatInputBox(self)
        self.input_box.return_pressed.connect(self._on_send_clicked)
        input_layout.addWidget(self.input_box, stretch=1)

        right_ctrl_layout = QVBoxLayout()
        right_ctrl_layout.setSpacing(6)

        top_opts_row = QHBoxLayout()
        top_opts_row.setSpacing(6)

        self.attach_btn = QPushButton("🖼️ 圖片")
        self.attach_btn.setObjectName("SecondaryBtn")
        self.attach_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.attach_btn.setToolTip("附加圖片進行 Vision 多模態對話 (需模型含 Vision Encoder)")
        self.attach_btn.clicked.connect(self._on_attach_image)
        top_opts_row.addWidget(self.attach_btn)

        self.think_checkbox = QCheckBox("💭 思考")
        self.think_checkbox.setChecked(False)
        self.think_checkbox.setToolTip(
            "勾選以啟用 Qwen3.8/3.6 <think> 深度推理模式；取消勾選則預填空白思考區塊以立即獲得高速回答"
        )
        top_opts_row.addWidget(self.think_checkbox)
        right_ctrl_layout.addLayout(top_opts_row)

        self.send_btn = QPushButton("發送 (Send)")
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.setFixedWidth(135)
        self.send_btn.setFixedHeight(36)
        self.send_btn.clicked.connect(self._on_send_clicked)
        right_ctrl_layout.addWidget(self.send_btn)

        input_layout.addLayout(right_ctrl_layout)
        bottom_layout.addWidget(input_frame)
        main_layout.addWidget(bottom_wrapper)

    def _setup_health_timer(self):
        self.health_timer = QTimer(self)
        self.health_timer.timeout.connect(self._check_health)
        self.health_timer.start(4000)
        self._check_health()

    def _check_health(self):
        if self.health_worker and self.health_worker.isRunning():
            return
        self.health_worker = HealthCheckWorker(self.config["base_url"])
        self.health_worker.status_updated.connect(self._on_health_updated)
        self.health_worker.start()

    def _on_health_updated(self, is_online: bool, data: dict):
        if is_online:
            device = data.get("device", "GPU")
            model = data.get("model", "Unknown")
            loaded = data.get("loaded", False)
            offload = data.get("offload_ratio", 0)
            gpu_mem = data.get("gpu_memory", {})
            sys_mem = data.get("system_memory", {})
            vram_mb = gpu_mem.get("allocated_memory_mb", 0.0)
            ram_avail = sys_mem.get("available_ram_gb", 0.0)

            if loaded:
                self.status_badge.setText(f"● {device} | {model[:28]}")
                self.status_badge.setProperty("online", "true")
            else:
                self.status_badge.setText(f"○ 已卸載顯存 ({device})")
                self.status_badge.setProperty("online", "true")

            self.strata_badge.setText(
                f"Offload: {offload}% | VRAM: {vram_mb:.0f} MB | RAM 可用: {ram_avail:.1f} GB"
            )

            # 更新下拉選單模型清單
            avail_models = data.get("available_models", [])
            if avail_models:
                existing = [self.model_combo.itemText(i) for i in range(self.model_combo.count())]
                new_ids = [m["id"] for m in avail_models]
                if existing != new_ids:
                    cur_sel = self.model_combo.currentText()
                    self.model_combo.blockSignals(True)
                    self.model_combo.clear()
                    self.model_combo.addItems(new_ids)
                    if model in new_ids:
                        self.model_combo.setCurrentText(model)
                    elif cur_sel in new_ids:
                        self.model_combo.setCurrentText(cur_sel)
                    self.model_combo.blockSignals(False)
        else:
            self.status_badge.setText("● 伺服器未連線 (Offline)")
            self.status_badge.setProperty("online", "false")
            self.strata_badge.setText("Offload: --% | VRAM: -- MB")

        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)

        if self.tray_manager:
            self.tray_manager._on_health_updated(is_online, data)

    def _on_load_selected_model(self):
        target_model = self.model_combo.currentText().strip()
        if not target_model:
            return
        if self.control_worker and self.control_worker.isRunning():
            return

        self.load_btn.setEnabled(False)
        self.unload_btn.setEnabled(False)
        self.status_badge.setText(f"⏳ 正在載入 {target_model[:22]}...")

        payload = {
            "model": target_model,
            "device": self.config["device"],
            "offload_ratio": self.config["offload_ratio"],
            "vram_budget_gb": self.config["vram_budget_gb"],
            "kv_cache_precision": self.config["kv_cache_precision"],
            "enable_mtp": self.config["enable_mtp"],
        }
        self.control_worker = ModelControlWorker(self.config["base_url"], "load", payload)
        self.control_worker.completed.connect(self._on_model_control_done)
        self.control_worker.start()

    def _on_unload_model(self):
        if self.control_worker and self.control_worker.isRunning():
            return
        self.load_btn.setEnabled(False)
        self.unload_btn.setEnabled(False)
        self.status_badge.setText("⏳ 正在釋放顯存...")

        self.control_worker = ModelControlWorker(self.config["base_url"], "unload")
        self.control_worker.completed.connect(self._on_model_control_done)
        self.control_worker.start()

    def _on_model_control_done(self, ok: bool, msg: str, _data: dict):
        self.load_btn.setEnabled(True)
        self.unload_btn.setEnabled(True)
        self._check_health()
        bubble = self.chat_history.add_message("assistant", f"{'✅' if ok else '⚠️'} **{msg}**")
        bubble.metrics_label.hide()

    def _on_attach_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇圖片檔案 (Vision Input)",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if file_path:
            self.attached_image_path = Path(file_path)
            self.attach_label.setText(f"🖼️ 已附加圖片: {self.attached_image_path.name}")
            self.attach_bar.show()

    def _clear_attached_image(self):
        self.attached_image_path = None
        self.attach_bar.hide()

    def _encode_attached_image_data_uri(self) -> Optional[str]:
        if not self.attached_image_path or not self.attached_image_path.exists():
            return None
        mime, _ = mimetypes.guess_type(str(self.attached_image_path))
        mime = mime or "image/png"
        b64 = base64.b64encode(self.attached_image_path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def _on_send_clicked(self):
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.stop()
            return

        user_text = self.input_box.toPlainText().strip()
        if not user_text and not self.attached_image_path:
            return

        self.input_box.clear()

        display_text = user_text
        if self.attached_image_path:
            display_text = f"🖼️ `[{self.attached_image_path.name}]`\n\n{user_text}"
        self.chat_history.add_message("user", display_text)

        if self.attached_image_path:
            data_uri = self._encode_attached_image_data_uri()
            content_payload: Any = [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": user_text or "Describe this image."},
            ]
            self._clear_attached_image()
        else:
            content_payload = user_text

        self.messages.append({"role": "user", "content": content_payload})

        req_messages: List[Dict[str, Any]] = []
        if self.config["system_prompt"]:
            req_messages.append({"role": "system", "content": self.config["system_prompt"]})
        req_messages.extend(self.messages)

        self.current_assistant_bubble = self.chat_history.add_message("assistant", "⏳ 思考與生成中...")
        self._set_generating_state(True)

        selected_model = self.model_combo.currentText().strip() or "default"
        self.current_worker = ChatStreamWorker(
            base_url=self.config["base_url"],
            messages=req_messages,
            model=selected_model,
            max_tokens=self.config["max_tokens"],
            temperature=self.config["temperature"],
            top_p=self.config["top_p"],
            enable_thinking=self.think_checkbox.isChecked(),
        )
        self._first_token = True
        self.current_worker.reasoning_received.connect(self._on_reasoning_received)
        self.current_worker.token_received.connect(self._on_token_received)
        self.current_worker.tool_calls_received.connect(self._on_tool_calls_received)
        self.current_worker.metrics_received.connect(self._on_metrics_received)
        self.current_worker.finished_signal.connect(self._on_stream_finished)
        self.current_worker.error_occurred.connect(self._on_stream_error)
        self.current_worker.start()

    def _on_reasoning_received(self, r_chunk: str):
        if not self.current_assistant_bubble:
            return
        if self._first_token:
            self.current_assistant_bubble.set_text("")
            self._first_token = False
        self.current_assistant_bubble.append_reasoning(r_chunk)
        self.chat_history.scroll_to_bottom()

    def _on_token_received(self, token: str):
        if not self.current_assistant_bubble:
            return
        if self._first_token:
            self.current_assistant_bubble.set_text("")
            self._first_token = False
        self.current_assistant_bubble.append_text(token)
        self.chat_history.scroll_to_bottom()

    def _on_tool_calls_received(self, tool_calls: list):
        if not self.current_assistant_bubble:
            return
        if self._first_token:
            self.current_assistant_bubble.set_text("")
            self._first_token = False
        self.current_assistant_bubble.append_tool_calls(tool_calls)
        self.chat_history.scroll_to_bottom()

    def _on_metrics_received(self, metrics: dict):
        if self.current_assistant_bubble:
            self.current_assistant_bubble.set_metrics(metrics)
            self.chat_history.scroll_to_bottom()

    def _on_stream_finished(self):
        if self.current_assistant_bubble:
            final_text = self.current_assistant_bubble.get_text()
            self.messages.append({"role": "assistant", "content": final_text})
        self._set_generating_state(False)
        self._check_health()

    def _on_stream_error(self, err_msg: str):
        if self.current_assistant_bubble:
            self.current_assistant_bubble.set_text(f"❌ **錯誤**:\n\n`{err_msg}`")
        self._set_generating_state(False)

    def _set_generating_state(self, is_generating: bool):
        if is_generating:
            self.send_btn.setText("⏹ 停止生成")
            self.send_btn.setObjectName("StopBtn")
        else:
            self.send_btn.setText("發送 (Send)")
            self.send_btn.setObjectName("")
        self.send_btn.style().unpolish(self.send_btn)
        self.send_btn.style().polish(self.send_btn)

    def _on_clear_chat(self):
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.stop()
        self.messages.clear()
        self.chat_history.clear_history()

    def _on_open_server_log(self):
        """開啟後端伺服器即時日誌檢視視窗"""
        if not hasattr(self, "_log_dialog") or self._log_dialog is None:
            self._log_dialog = ServerLogDialog(base_url=self.config["base_url"], parent=self)
        self._log_dialog.show()
        self._log_dialog.raise_()
        self._log_dialog.activateWindow()

    def _on_open_settings(self):
        dialog = SettingsDialog(self.config, self)
        if dialog.exec():
            updated = dialog.get_updated_config()
            apply_now = updated.pop("apply_strata_on_save", False)
            if "minimize_to_tray" in updated:
                self.minimize_to_tray_on_close = updated["minimize_to_tray"]
                if self.tray_manager:
                    self.tray_manager.minimize_on_close = self.minimize_to_tray_on_close
                    self.tray_manager.action_minimize_toggle.setChecked(self.minimize_to_tray_on_close)
            if "autostart_on_login" in updated:
                from src.utils.autostart import set_autostart, is_autostart_enabled
                if updated["autostart_on_login"] != is_autostart_enabled():
                    set_autostart(updated["autostart_on_login"])
            self.config.update(updated)
            self._check_health()
            if apply_now:
                self._on_load_selected_model()

    def _save_window_state(self):
        """儲存上次關閉或隱藏時的視窗幾何座標、尺寸與最大化狀態"""
        if not self.property("_has_been_shown"):
            return

        is_max = self.isMaximized()
        is_min = self.isMinimized()

        if is_max or is_min:
            normal_rect = self.normalGeometry()
            if normal_rect.isValid() and normal_rect.width() >= self.minimumWidth():
                rect = normal_rect
            else:
                rect = self.geometry()
        else:
            rect = self.geometry()

        if rect.width() < self.minimumWidth() or rect.height() < self.minimumHeight():
            return

        settings = QSettings("OpenVINO-Strata", "AIStudio")
        settings.setValue("window/x", rect.x())
        settings.setValue("window/y", rect.y())
        settings.setValue("window/width", rect.width())
        settings.setValue("window/height", rect.height())
        settings.setValue("window/maximized", is_max)
        settings.setValue("window/has_saved_state", True)
        settings.sync()

    def _is_rect_visible_on_screens(self, target_rect: QRect, screens_list) -> bool:
        """檢查視窗矩形是否安全落在目前連接的任一螢幕可用區域內，且頂部標題列可正常操作"""
        if not screens_list or not target_rect.isValid():
            return False

        min_visible_w = 200
        min_visible_h = 100

        for s in screens_list:
            s_avail = s.availableGeometry()
            inter = target_rect.intersected(s_avail)
            # 至少有 200x100 的視窗主體落在該螢幕可用區域內
            if inter.width() >= min_visible_w and inter.height() >= min_visible_h:
                # 檢查視窗頂端標題列 (前 35px) 是否有至少 100px 寬與 15px 高在螢幕內供使用者拖曳抓取
                title_rect = QRect(
                    target_rect.x(), target_rect.y(),
                    target_rect.width(), min(35, target_rect.height())
                )
                title_inter = title_rect.intersected(s_avail)
                if title_inter.width() >= 100 and title_inter.height() >= 15:
                    return True
        return False

    def _place_on_first_screen(self, first_screen, preferred_w: int, preferred_h: int, is_max: bool):
        """將視窗安全置中於第一個螢幕 (Screen 0 / 主螢幕)"""
        screens = QGuiApplication.screens()
        if not first_screen:
            first_screen = screens[0] if screens else QGuiApplication.primaryScreen()
        if not first_screen:
            return

        avail = first_screen.availableGeometry()
        w = min(preferred_w, avail.width() - 40)
        w = max(self.minimumWidth(), w)
        h = min(preferred_h, avail.height() - 40)
        h = max(self.minimumHeight(), h)

        x = avail.x() + (avail.width() - w) // 2
        y = avail.y() + (avail.height() - h) // 2
        self.setGeometry(QRect(x, y, w, h))

        if is_max:
            self.setWindowState(self.windowState() | Qt.WindowState.WindowMaximized)

    def _restore_window_state(self):
        """啟動時還原上次關閉時的位置；若超出顯示區域 (如拔除外接螢幕)，則自動顯示在第一個螢幕"""
        screens = QGuiApplication.screens()
        first_screen = screens[0] if screens else QGuiApplication.primaryScreen()

        settings = QSettings("OpenVINO-Strata", "AIStudio")
        has_saved = settings.value("window/has_saved_state", False, type=bool)

        if not has_saved:
            # 初次啟動：預設置中於第一個螢幕
            self._place_on_first_screen(first_screen, 1120, 780, is_max=False)
            return

        saved_x = settings.value("window/x", type=int)
        saved_y = settings.value("window/y", type=int)
        saved_w = settings.value("window/width", 1120, type=int)
        saved_h = settings.value("window/height", 780, type=int)
        is_max = settings.value("window/maximized", False, type=bool)

        saved_w = max(self.minimumWidth(), saved_w)
        saved_h = max(self.minimumHeight(), saved_h)
        target_rect = QRect(saved_x, saved_y, saved_w, saved_h)

        if self._is_rect_visible_on_screens(target_rect, screens):
            # 位置安全：直接還原儲存的幾何位置
            self.setGeometry(target_rect)
            if is_max:
                self.setWindowState(self.windowState() | Qt.WindowState.WindowMaximized)
        else:
            # 超出顯示區域：自動安全還原至第一個螢幕
            self._place_on_first_screen(first_screen, saved_w, saved_h, is_max)

    def showEvent(self, event):
        super().showEvent(event)
        self.setProperty("_has_been_shown", True)
        # 喚醒或還原時再次安全校驗：防止於常駐系統匣期間外接螢幕被拔除
        screens = QGuiApplication.screens()
        if screens and not self._is_rect_visible_on_screens(self.geometry(), screens):
            first_screen = screens[0]
            self._place_on_first_screen(
                first_screen,
                preferred_w=self.width(),
                preferred_h=self.height(),
                is_max=self.isMaximized(),
            )

    def hideEvent(self, event):
        super().hideEvent(event)
        self._save_window_state()

    def closeEvent(self, event):
        """攔截視窗關閉事件：儲存視窗位置，若啟用系統匣最小化則隱藏並退至系統匣"""
        self._save_window_state()
        if (
            self.minimize_to_tray_on_close
            and self.tray_manager
            and hasattr(self.tray_manager, "tray_icon")
            and self.tray_manager.tray_icon.isVisible()
        ):
            event.ignore()
            self.hide()
            self.tray_manager.on_window_minimized_to_tray()
        else:
            if hasattr(self, "health_timer") and self.health_timer.isActive():
                self.health_timer.stop()
            if self.current_worker and self.current_worker.isRunning():
                self.current_worker.stop()
            event.accept()
