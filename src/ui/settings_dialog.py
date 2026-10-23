"""
參數與 PolyStrata 顯存分層設定對話框 (src/ui/settings_dialog.py)
提供：
1. 伺服器連線設定 (Base URL)
2. 系統提示詞 (System Prompt)
3. 生成超參數 (Temperature, Top-P, Max New Tokens)
4. PolyStrata 分層記憶體與 MoE 卸載設定 (Target Device, VRAM Budget GB, Offload Ratio %, KV Precision, MTP)
"""

from typing import Dict, Any

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QSlider,
    QComboBox,
    QCheckBox,
    QLabel,
    QPushButton,
    QPlainTextEdit,
)
from PyQt6.QtCore import Qt


class SettingsDialog(QDialog):
    def __init__(self, current_config: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ OpenVINO-Strata 推論與顯存分層設定")
        self.setMinimumWidth(520)
        self.config = current_config.copy()
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 18, 20, 18)

        # 1. 伺服器連線設定
        conn_group = QGroupBox("連線設定 (Server Connection)")
        conn_layout = QFormLayout(conn_group)
        self.url_input = QLineEdit(self.config.get("base_url", "http://127.0.0.1:12223"))
        conn_layout.addRow("API 位址:", self.url_input)
        layout.addWidget(conn_group)

        # 2. PolyStrata 顯存分層與 MoE 卸載設定
        strata_group = QGroupBox("PolyStrata 顯存分層設定 (35B-A3B Low-VRAM Tiering)")
        strata_layout = QFormLayout(strata_group)

        self.device_combo = QComboBox()
        # 35B MoE 主模型必須在 GPU (透過 OFFLOAD_RATIO 卸載) 或 CPU 執行；
        # NPU 不支援 OFFLOAD_RATIO，AUTO 會造成重複編譯，故從主模型選單中排除
        devices = ["GPU", "CPU", "HETERO:GPU,CPU"]
        self.device_combo.addItems(devices)
        cur_dev = self.config.get("device", "GPU")
        if cur_dev in devices:
            self.device_combo.setCurrentText(cur_dev)
        else:
            self.device_combo.setCurrentText("GPU")
        self.device_combo.setToolTip(
            "35B-A3B MoE 主模型仰賴 OFFLOAD_RATIO 分層卸載，僅支援 GPU 或 CPU。\n"
            "NPU 不支援動態專家卸載，但可作為副引擎 (Sub-Engine) 處理小模型/快回請求。"
        )
        strata_layout.addRow("主模型裝置 (Device):", self.device_combo)

        self.vram_budget_spin = QDoubleSpinBox()
        self.vram_budget_spin.setRange(0.0, 64.0)
        self.vram_budget_spin.setSingleStep(0.5)
        self.vram_budget_spin.setDecimals(1)
        self.vram_budget_spin.setSpecialValueText("自動偵測硬體上限 (Auto)")
        self.vram_budget_spin.setSuffix(" GB")
        self.vram_budget_spin.setValue(float(self.config.get("vram_budget_gb") or 0.0))
        strata_layout.addRow("VRAM 預算上限:", self.vram_budget_spin)

        self.offload_spin = QSpinBox()
        self.offload_spin.setRange(-1, 100)
        self.offload_spin.setSingleStep(5)
        self.offload_spin.setSpecialValueText("依 VRAM 預算自動試算 (-1)")
        self.offload_spin.setSuffix(" %")
        self.offload_spin.setValue(int(self.config.get("offload_ratio", -1)))
        strata_layout.addRow("MoE 專家卸載率 (OFFLOAD_RATIO):", self.offload_spin)

        self.kv_combo = QComboBox()
        self.kv_combo.addItems(["u8", "f16"])
        self.kv_combo.setCurrentText(self.config.get("kv_cache_precision", "u8"))
        strata_layout.addRow("KV Cache 精度:", self.kv_combo)

        self.mtp_check = QCheckBox("啟用 Multi-Token Prediction (MTP 推測解碼，若模型支援)")
        self.mtp_check.setChecked(bool(self.config.get("enable_mtp", True)))
        strata_layout.addRow("", self.mtp_check)

        self.apply_strata_check = QCheckBox("儲存時立即套用 Strata 設定並重新載入模型")
        self.apply_strata_check.setChecked(False)
        strata_layout.addRow("", self.apply_strata_check)

        layout.addWidget(strata_group)

        # 3. 系統提示詞
        sys_group = QGroupBox("系統提示詞 (System Prompt)")
        sys_layout = QVBoxLayout(sys_group)
        self.sys_prompt_input = QPlainTextEdit(
            self.config.get("system_prompt", "You are a helpful AI assistant.")
        )
        self.sys_prompt_input.setFixedHeight(72)
        self.sys_prompt_input.setStyleSheet(
            "background-color: #1E232D; border: 1px solid #2E3646; border-radius: 6px; padding: 6px; color: #F8FAFC;"
        )
        sys_layout.addWidget(self.sys_prompt_input)
        layout.addWidget(sys_group)

        # 4. 生成超參數
        gen_group = QGroupBox("生成超參數 (Generation Hyperparameters)")
        gen_layout = QFormLayout(gen_group)

        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(16, 16384)
        self.max_tokens_spin.setSingleStep(128)
        self.max_tokens_spin.setValue(int(self.config.get("max_tokens", 1024)))
        gen_layout.addRow("Max New Tokens:", self.max_tokens_spin)

        # Temperature
        temp_val = float(self.config.get("temperature", 0.7))
        temp_row = QHBoxLayout()
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(0, 150)
        self.temp_slider.setValue(int(temp_val * 100))
        self.temp_label = QLabel(f"{temp_val:.2f}")
        self.temp_label.setFixedWidth(40)
        self.temp_slider.valueChanged.connect(lambda v: self.temp_label.setText(f"{v / 100:.2f}"))
        temp_row.addWidget(self.temp_slider)
        temp_row.addWidget(self.temp_label)
        gen_layout.addRow("Temperature:", temp_row)

        # Top-P
        top_p_val = float(self.config.get("top_p", 0.9))
        top_p_row = QHBoxLayout()
        self.top_p_slider = QSlider(Qt.Orientation.Horizontal)
        self.top_p_slider.setRange(5, 100)
        self.top_p_slider.setValue(int(top_p_val * 100))
        self.top_p_label = QLabel(f"{top_p_val:.2f}")
        self.top_p_label.setFixedWidth(40)
        self.top_p_slider.valueChanged.connect(lambda v: self.top_p_label.setText(f"{v / 100:.2f}"))
        top_p_row.addWidget(self.top_p_slider)
        top_p_row.addWidget(self.top_p_label)
        gen_layout.addRow("Top-P:", top_p_row)

        layout.addWidget(gen_group)

        # 5. 系統匣與桌面行為
        tray_group = QGroupBox("系統匣與桌面行為 (Windows System Tray & Startup)")
        tray_layout = QVBoxLayout(tray_group)
        self.minimize_to_tray_check = QCheckBox("視窗按關閉 (✕) 時最小化至 Windows 系統匣常駐運作")
        self.minimize_to_tray_check.setChecked(bool(self.config.get("minimize_to_tray", True)))
        tray_layout.addWidget(self.minimize_to_tray_check)

        from src.utils.autostart import is_autostart_supported, is_autostart_enabled
        self.autostart_check = QCheckBox("Windows 使用者登入後自動在背景啟動 (靜默無黑窗模式)")
        self.autostart_check.setChecked(is_autostart_enabled())
        self.autostart_check.setEnabled(is_autostart_supported())
        tray_layout.addWidget(self.autostart_check)

        layout.addWidget(tray_group)

        # 6. 底部按鈕
        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)

        cancel_btn = QPushButton("取消")
        cancel_btn.setObjectName("SecondaryBtn")
        cancel_btn.clicked.connect(self.reject)

        save_btn = QPushButton("儲存設定")
        save_btn.clicked.connect(self.accept)

        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

    def get_updated_config(self) -> Dict[str, Any]:
        vram_val = self.vram_budget_spin.value()
        return {
            "base_url": self.url_input.text().strip(),
            "system_prompt": self.sys_prompt_input.toPlainText().strip(),
            "max_tokens": self.max_tokens_spin.value(),
            "temperature": round(self.temp_slider.value() / 100.0, 2),
            "top_p": round(self.top_p_slider.value() / 100.0, 2),
            "device": self.device_combo.currentText(),
            "vram_budget_gb": vram_val if vram_val > 0.0 else None,
            "offload_ratio": self.offload_spin.value(),
            "kv_cache_precision": self.kv_combo.currentText(),
            "enable_mtp": self.mtp_check.isChecked(),
            "apply_strata_on_save": self.apply_strata_check.isChecked(),
            "minimize_to_tray": self.minimize_to_tray_check.isChecked(),
            "autostart_on_login": self.autostart_check.isChecked(),
        }
