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

    window = MainWindow(base_url=args.url)
    window.show()

    if args.smoke_test:
        window.close()
        return 0

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
