"""
OpenVINO-Strata Windows 系統匣常駐模式啟動入口 (run_tray.py)
功能特點：
1. 啟動並常駐於 Windows 工作列右下角系統匣 (System Tray)
2. 自動偵測後端服務狀態，若未啟動可自動於背景啟動 run_server.py
3. 單擊/雙擊系統匣圖示秒開/喚醒 Chat UI
4. 右鍵選單支援：開啟 Chat UI、開啟 Swagger API 文件、即時顯存監控、一鍵切換模型、一鍵釋放顯存 (Unload VRAM)、偏好設定與完整退出
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
from PyQt6.QtWidgets import QApplication

from src.ui.main_window import MainWindow
from src.ui.styles import DARK_THEME_QSS
from src.ui.tray_manager import StrataTrayManager, get_app_icon
from src.utils.autostart import set_autostart, is_autostart_enabled, get_autostart_status_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ov_strata.tray_app")


def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenVINO-Strata Windows System Tray Mode Runner"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:12223",
        help="後端 API 伺服器 URL (預設 http://127.0.0.1:12223)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="自動啟動後端伺服器時的監聽主機位址 (預設 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=12223,
        help="自動啟動後端伺服器時的監聽連接埠 (預設 12223)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="GPU",
        help="目標運算裝置 (GPU/NPU/CPU，預設 GPU)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="啟動時預先載入之模型名稱或目錄路徑",
    )
    parser.add_argument(
        "--vram-budget",
        type=float,
        default=None,
        help="最大 GPU 顯存預算 (GB)",
    )
    parser.add_argument(
        "--start-server",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="若後端未啟動，自動於背景啟動 run_server.py (預設啟用，使用 --no-start-server 可停用)",
    )
    parser.add_argument(
        "--minimized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="啟動時直接最小化常駐於系統匣 (預設 True，使用 --no-minimized 可立即開啟 Chat UI)",
    )
    parser.add_argument(
        "--install-startup",
        action="store_true",
        help="安裝為 Windows 登入自動在背景啟動 (靜默無黑窗模式)",
    )
    parser.add_argument(
        "--uninstall-startup",
        action="store_true",
        help="解除 Windows 登入自動啟動",
    )
    parser.add_argument(
        "--status-startup",
        action="store_true",
        help="檢視 Windows 登入自動啟動狀態",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="初始化系統匣元件與主視窗後立即正常結束 (供自動化測試)",
    )
    return parser.parse_args()


def is_server_alive(base_url: str) -> bool:
    try:
        with httpx.Client(timeout=1.5) as client:
            resp = client.get(f"{base_url.rstrip('/')}/health")
            return resp.status_code == 200
    except Exception:
        return False


def start_background_server(args) -> subprocess.Popen:
    """以無主控台視窗的背景程序模式啟動 run_server.py，並將日誌持續導流至 logs/server.log"""
    project_root = Path(__file__).resolve().parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    server_log_path = logs_dir / "server.log"

    cmd = [
        sys.executable,
        str(project_root / "run_server.py"),
        "--host",
        str(args.host),
        "--port",
        str(args.port),
        "--device",
        str(args.device),
    ]
    if args.model:
        cmd.extend(["--model", str(args.model)])
    if args.vram_budget:
        cmd.extend(["--vram-budget", str(args.vram_budget)])

    creationflags = 0
    if sys.platform == "win32":
        # Windows: CREATE_NO_WINDOW 避免彈出黑色命令提示字元視窗
        creationflags = subprocess.CREATE_NO_WINDOW

    logger.info("啟動背景 OpenVINO-Strata API 伺服器 (日誌寫入 %s): %s", server_log_path, " ".join(cmd))
    log_fp = open(server_log_path, "a", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        cmd,
        cwd=str(project_root),
        creationflags=creationflags,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
    )
    # 將 log 檔案指標附加於 proc，防止垃圾回收過早關閉
    proc._log_file_handle = log_fp
    return proc


def main() -> int:
    args = parse_args()

    # 處理 Windows 登入自動啟動命令列指令
    if args.install_startup:
        ok, msg = set_autostart(True)
        print(f"[{'OK' if ok else 'ERROR'}] {msg}")
        return 0 if ok else 1

    if args.uninstall_startup:
        ok, msg = set_autostart(False)
        print(f"[{'OK' if ok else 'ERROR'}] {msg}")
        return 0 if ok else 1

    if args.status_startup:
        info = get_autostart_status_info()
        print("=" * 60)
        print("  OpenVINO-Strata Windows 登入自啟動狀態")
        print("=" * 60)
        print(f"[*] 支援自啟: {info['supported']}")
        print(f"[*] 目前狀態: {'已啟用 (Enabled)' if info['enabled'] else '未啟用 (Disabled)'}")
        print(f"[*] 啟動指令: {info['command']}")
        print(f"[*] 註冊路徑: {info['reg_path']}")
        print("=" * 60)
        return 0

    app = QApplication(sys.argv)
    app.setApplicationName("OpenVINO-Strata AI Studio")
    app.setStyleSheet(DARK_THEME_QSS)
    app.setQuitOnLastWindowClosed(False)

    server_proc: subprocess.Popen | None = None
    url = args.url.rstrip("/")

    # 若需要且伺服器未運行，啟動背景伺服器
    if args.start_server and not is_server_alive(url) and not args.smoke_test:
        server_proc = start_background_server(args)

    window = MainWindow(
        base_url=url,
        enable_tray=True,
        minimize_on_close=True,
    )

    # 若有託管的 server 程序，注入 tray_manager 進行生命週期管理
    if server_proc and window.tray_manager:
        window.tray_manager.server_process = server_proc

    if not args.minimized:
        window.show()
    else:
        window.hide()
        if window.tray_manager:
            if server_proc:
                window.tray_manager.show_notification(
                    "OpenVINO-Strata 系統匣模式",
                    "伺服器已在背景啟動中，Chat UI 已常駐於系統匣。\n按一下系統匣圖示即可隨時開啟聊天室。",
                )
            else:
                window.tray_manager.show_notification(
                    "OpenVINO-Strata 已常駐於系統匣",
                    "應用程式已就緒！按一下工作列右下角圖示即可開啟 Chat UI。",
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
