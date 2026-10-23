"""
現代化深色主題 QSS 樣式表 (Modern Dark Theme for OpenVINO-Strata)
針對原生 PyQt6 桌面元件進行美化，包含 <think> 推理摺疊面板、Strata 顯存狀態膠囊與模型熱切換控制項。
"""

DARK_THEME_QSS = """
/* 全域基礎設定 */
QWidget {
    background-color: #121316;
    color: #E2E8F0;
    font-family: "Segoe UI", "Microsoft JhengHei", sans-serif;
    font-size: 14px;
}

QMainWindow, QDialog {
    background-color: #121316;
}

/* 頂部標題與狀態列容器 */
#HeaderFrame {
    background-color: #1A1D24;
    border-bottom: 1px solid #282D37;
    padding: 8px 16px;
}

#StatusBadge {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 500;
}

#StatusBadge[online="true"] {
    color: #10B981;
    border-color: #059669;
    background-color: rgba(16, 185, 129, 0.12);
}

#StatusBadge[online="false"] {
    color: #EF4444;
    border-color: #DC2626;
    background-color: rgba(239, 68, 68, 0.12);
}

#StrataBadge {
    background-color: rgba(59, 130, 246, 0.12);
    border: 1px solid #2563EB;
    border-radius: 12px;
    padding: 4px 10px;
    font-size: 12px;
    color: #93C5FD;
    font-weight: 500;
}

/* 對話瀏覽區域 */
QScrollArea {
    border: none;
    background-color: #121316;
}

#ChatContainer {
    background-color: #121316;
}

/* <think> 深度推理摺疊框 */
#ThinkFrame {
    background-color: #141821;
    border-left: 3px solid #8B5CF6;
    border-radius: 6px;
    padding: 6px 10px;
}

/* 輸入框卡片 */
#InputFrame {
    background-color: #1A1D24;
    border: 1px solid #282D37;
    border-radius: 14px;
    padding: 8px 12px;
}

#InputFrame:focus-within {
    border-color: #3B82F6;
}

QPlainTextEdit#ChatInput {
    background-color: transparent;
    border: none;
    color: #F8FAFC;
    font-size: 14px;
    line-height: 1.5;
}

/* 按鈕樣式 */
QPushButton {
    background-color: #2563EB;
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    padding: 7px 14px;
    font-weight: 600;
    font-size: 13px;
}

QPushButton:hover {
    background-color: #1D4ED8;
}

QPushButton:pressed {
    background-color: #1E40AF;
}

QPushButton:disabled {
    background-color: #334155;
    color: #64748B;
}

/* 次要 / 功能按鈕 */
QPushButton#SecondaryBtn {
    background-color: #1E232D;
    color: #CBD5E1;
    border: 1px solid #2E3646;
}

QPushButton#SecondaryBtn:hover {
    background-color: #283040;
    border-color: #3B82F6;
    color: #FFFFFF;
}

QPushButton#UnloadBtn {
    background-color: #1E232D;
    color: #F59E0B;
    border: 1px solid #92400E;
}

QPushButton#UnloadBtn:hover {
    background-color: rgba(245, 158, 11, 0.15);
    border-color: #F59E0B;
    color: #FDE68A;
}

QPushButton#StopBtn {
    background-color: #DC2626;
    color: #FFFFFF;
}

QPushButton#StopBtn:hover {
    background-color: #B91C1C;
}

/* 設定對話框分組框 */
QGroupBox {
    font-weight: bold;
    border: 1px solid #282D37;
    border-radius: 10px;
    margin-top: 14px;
    padding: 16px 12px 12px 12px;
    background-color: #181B22;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: #60A5FA;
}

/* 標籤 */
QLabel {
    color: #94A3B8;
}

QLabel#TitleLabel {
    font-size: 16px;
    font-weight: bold;
    color: #F8FAFC;
}

/* 輸入控制項 */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #1E232D;
    border: 1px solid #2E3646;
    border-radius: 6px;
    padding: 5px 10px;
    color: #F8FAFC;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #3B82F6;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QCheckBox {
    color: #CBD5E1;
    spacing: 6px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #475569;
    background-color: #1E232D;
}

QCheckBox::indicator:checked {
    background-color: #2563EB;
    border-color: #3B82F6;
}

/* 滑動條 (QSlider) */
QSlider::groove:horizontal {
    border: none;
    height: 4px;
    background: #334155;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: #3B82F6;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #60A5FA;
    border: 2px solid #1E293B;
    width: 14px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #93C5FD;
}

/* 捲軸 (QScrollBar) */
QScrollBar:vertical {
    border: none;
    background: #121316;
    width: 8px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #2D3748;
    min-height: 24px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #4A5568;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* 系統匣與快顯選單 (QMenu) */
QMenu {
    background-color: #1A1D24;
    border: 1px solid #282D37;
    border-radius: 8px;
    padding: 6px 4px;
    color: #E2E8F0;
    font-size: 13px;
}

QMenu::item {
    background-color: transparent;
    padding: 6px 22px 6px 14px;
    border-radius: 5px;
    margin: 1px 4px;
}

QMenu::item:selected {
    background-color: #2563EB;
    color: #FFFFFF;
}

QMenu::item:disabled {
    color: #64748B;
    background-color: transparent;
}

QMenu::separator {
    height: 1px;
    background-color: #282D37;
    margin: 5px 8px;
}

QMenu::indicator {
    width: 14px;
    height: 14px;
    margin-left: 4px;
}
"""
