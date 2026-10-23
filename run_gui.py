"""
OpenVINO-Strata 原生桌面聊天室啟動入口 (run_gui.py)
支援指令列參數：
  python run_gui.py --url http://127.0.0.1:8000
"""

import argparse
import sys

from PyQt6.QtWidgets import QApplication

from src.ui.main_window import MainWindow
from src.ui.styles import DARK_THEME_QSS


def parse_args():
    parser = argparse.ArgumentParser(description="OpenVINO-Strata Desktop Chat GUI")
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:12223",
        help="Backend API Server Base URL (default: http://127.0.0.1:12223)",
    )
    parser.add_argument(
        "--tray",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="啟用 Windows 系統匣常駐圖示 (預設啟用，使用 --no-tray 可關閉)",
    )
    parser.add_argument(
        "--minimized",
        "--start-in-tray",
        dest="minimized",
        action="store_true",
        default=False,
        help="啟動時直接最小化常駐於 Windows 系統匣，暫不開啟主視窗",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Instantiate GUI widgets and exit immediately (for automated verification)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("OpenVINO-Strata AI Studio")
    app.setStyleSheet(DARK_THEME_QSS)

    if args.tray:
        app.setQuitOnLastWindowClosed(False)

    window = MainWindow(base_url=args.url, enable_tray=args.tray)

    if not args.minimized:
        window.show()
    elif window.tray_manager:
        window.tray_manager.show_notification(
            "OpenVINO-Strata 已就緒",
            "應用程式已啟動並常駐於系統匣。\n按一下系統匣圖示即可隨時開啟 Chat UI。",
        )

    if args.smoke_test:
        if window.tray_manager:
            window.tray_manager.quit_application()
        else:
            window.close()
        return 0

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
