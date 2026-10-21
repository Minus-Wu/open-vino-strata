"""
單一對話氣泡與捲動歷史區域元件 (src/ui/chat_widget.py)
支援：
1. Markdown 富文本與程式碼區塊渲染
2. <think>...</think> (`reasoning_content`) 可摺疊深度思考面板
3. <tool_call> 結構化工具呼叫展示
4. Strata 分層記憶體與生成效能指標頁尾 (Decode tok/s、TTFT、Offload %、VRAM MB)
"""

import html
import json
from typing import Dict, Any, List

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QFrame,
    QSizePolicy,
    QPushButton,
)
from PyQt6.QtCore import Qt, QTimer


class MessageBubble(QFrame):
    def __init__(self, role: str, text: str = "", parent=None):
        super().__init__(parent)
        self.role = role
        self._raw_text = text
        self._reasoning_text = ""
        self._tool_calls: List[Dict[str, Any]] = []
        self._think_expanded = True
        self._setup_ui()

    def _setup_ui(self):
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        outer_layout = QHBoxLayout(self)
        outer_layout.setContentsMargins(12, 6, 12, 6)

        self.card = QFrame(self)
        self.card.setMaximumWidth(860)
        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(16, 12, 16, 12)
        card_layout.setSpacing(8)

        # 角色標頭
        header_label = QLabel()
        header_label.setStyleSheet(
            "font-size: 12px; font-weight: bold; background: transparent; border: none;"
        )

        if self.role == "user":
            header_label.setText("👤 您 (User)")
            header_label.setStyleSheet(header_label.styleSheet() + "color: #93C5FD;")
            self.card.setStyleSheet(
                """
                QFrame {
                    background-color: #1E3A8A;
                    border: 1px solid #2563EB;
                    border-radius: 14px;
                }
                """
            )
            outer_layout.addStretch(1)
            outer_layout.addWidget(self.card, stretch=4)
        else:
            header_label.setText("🤖 OpenVINO-Strata Assistant")
            header_label.setStyleSheet(header_label.styleSheet() + "color: #34D399;")
            self.card.setStyleSheet(
                """
                QFrame {
                    background-color: #1E222B;
                    border: 1px solid #2E3646;
                    border-radius: 14px;
                }
                """
            )
            outer_layout.addWidget(self.card, stretch=4)
            outer_layout.addStretch(1)

        card_layout.addWidget(header_label)

        # 可摺疊 <think> 思考鏈面板 (預設隱藏，收到 reasoning_content 時自動顯示)
        self.think_container = QFrame()
        self.think_container.setObjectName("ThinkFrame")
        self.think_container.setStyleSheet(
            """
            QFrame#ThinkFrame {
                background-color: #141821;
                border-left: 3px solid #8B5CF6;
                border-radius: 6px;
            }
            """
        )
        think_layout = QVBoxLayout(self.think_container)
        think_layout.setContentsMargins(10, 6, 10, 6)
        think_layout.setSpacing(4)

        self.think_toggle_btn = QPushButton("💭 深度思考過程 (Reasoning Chain) ▼")
        self.think_toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.think_toggle_btn.setStyleSheet(
            """
            QPushButton {
                background: transparent;
                border: none;
                color: #A78BFA;
                font-size: 12px;
                font-weight: bold;
                text-align: left;
                padding: 2px 0px;
            }
            QPushButton:hover {
                color: #C4B5FD;
            }
            """
        )
        self.think_toggle_btn.clicked.connect(self._toggle_thinking_view)
        think_layout.addWidget(self.think_toggle_btn)

        self.think_label = QLabel("")
        self.think_label.setWordWrap(True)
        self.think_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.think_label.setStyleSheet(
            "color: #94A3B8; font-size: 13px; font-style: italic; background: transparent; border: none;"
        )
        think_layout.addWidget(self.think_label)
        self.think_container.hide()
        card_layout.addWidget(self.think_container)

        # 內文區域 (支援 Markdown 與文字選取複製)
        self.content_label = QLabel(self._raw_text)
        self.content_label.setWordWrap(True)
        self.content_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        self.content_label.setTextFormat(Qt.TextFormat.MarkdownText)
        self.content_label.setOpenExternalLinks(True)
        self.content_label.setStyleSheet(
            "font-size: 14px; color: #F1F5F9; background: transparent; border: none; line-height: 1.5;"
        )
        card_layout.addWidget(self.content_label)

        # Tool Calls 標籤 (預設隱藏)
        self.tool_label = QLabel("")
        self.tool_label.setWordWrap(True)
        self.tool_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.tool_label.setStyleSheet(
            """
            font-size: 12px;
            font-family: Consolas, monospace;
            color: #FCD34D;
            background-color: #171B24;
            border: 1px solid #92400E;
            border-radius: 6px;
            padding: 6px 10px;
            """
        )
        self.tool_label.hide()
        card_layout.addWidget(self.tool_label)

        # 效能與 Strata 顯存指標小字 (僅 Assistant 顯示)
        self.metrics_label = QLabel("")
        self.metrics_label.setStyleSheet(
            "font-size: 11px; color: #94A3B8; background: transparent; border: none; margin-top: 4px;"
        )
        self.metrics_label.hide()
        card_layout.addWidget(self.metrics_label)

    def _toggle_thinking_view(self):
        self._think_expanded = not self._think_expanded
        self.think_label.setVisible(self._think_expanded)
        arrow = "▼" if self._think_expanded else "▶"
        self.think_toggle_btn.setText(f"💭 深度思考過程 (Reasoning Chain) {arrow}")

    def append_reasoning(self, chunk: str):
        self._reasoning_text += chunk
        if self.think_container.isHidden():
            self.think_container.show()
        self.think_label.setText(self._reasoning_text.strip())

    def append_text(self, chunk: str):
        self._raw_text += chunk
        self.content_label.setText(self._raw_text)

    def append_tool_calls(self, tool_calls: List[Dict[str, Any]]):
        self._tool_calls.extend(tool_calls)
        lines = []
        for tc in self._tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", tc.get("name", "tool"))
            args = fn.get("arguments", "")
            lines.append(f"🔧 Tool Call: {name}({args})")
        if lines:
            self.tool_label.setText("\n".join(lines))
            self.tool_label.show()

    def set_text(self, text: str):
        self._raw_text = text
        self.content_label.setText(self._raw_text)

    def get_text(self) -> str:
        return self._raw_text

    def set_metrics(self, metrics: Dict[str, Any]):
        tps = metrics.get("tps", 0)
        ttft = metrics.get("ttft", 0)
        tokens = metrics.get("tokens", 0)
        r_tokens = metrics.get("reasoning_tokens", 0)
        total_time = metrics.get("total_time", 0)
        offload = metrics.get("offload_ratio", 0)
        vram_mb = metrics.get("vram_mb", 0.0)

        parts = [
            f"⚡ {tps} tok/s",
            f"⏱ TTFT: {ttft}s",
            f"🔢 {tokens} tokens" + (f" (💭 {r_tokens})" if r_tokens else ""),
            f"⏳ {total_time}s",
        ]
        if offload > 0:
            parts.append(f"🧠 Offload: {offload}%")
        if vram_mb > 0:
            parts.append(f"🎮 VRAM: {vram_mb:.0f} MB")

        self.metrics_label.setText("  |  ".join(parts))
        self.metrics_label.show()


class ChatHistoryWidget(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.container = QWidget()
        self.container.setObjectName("ChatContainer")
        self.layout = QVBoxLayout(self.container)
        self.layout.setContentsMargins(8, 12, 8, 12)
        self.layout.setSpacing(10)
        self.layout.addStretch(1)

        self.setWidget(self.container)
        self.bubbles: List[MessageBubble] = []

    def add_message(self, role: str, text: str = "") -> MessageBubble:
        bubble = MessageBubble(role=role, text=text, parent=self.container)
        self.layout.insertWidget(self.layout.count() - 1, bubble)
        self.bubbles.append(bubble)
        self.scroll_to_bottom()
        return bubble

    def clear_history(self):
        for bubble in self.bubbles:
            self.layout.removeWidget(bubble)
            bubble.deleteLater()
        self.bubbles.clear()

    def scroll_to_bottom(self):
        QTimer.singleShot(
            15,
            lambda: self.verticalScrollBar().setValue(self.verticalScrollBar().maximum()),
        )
