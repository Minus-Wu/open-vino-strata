"""
Windows 登入自動啟動 (Autostart on Logon) 管理模組 (src/utils/autostart.py)
利用 Windows 註冊表 HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run
支援：
1. 查詢是否已設定開機/登入自動常駐啟動
2. 一鍵啟用自動啟動 (使用 pythonw.exe 靜默背景啟動 run_tray.py --minimized，無黑窗)
3. 一鍵停用自動啟動
"""

import os
import sys
from pathlib import Path
from typing import Tuple

REG_KEY_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"
APP_NAME = "OpenVINO-Strata"


def is_autostart_supported() -> bool:
    """判斷當前作業系統是否支援此註冊表自啟機制"""
    return sys.platform == "win32"


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def get_autostart_command() -> str:
    """取得靜默背景啟動命令列字串 (使用 pythonw.exe 避免彈出主控台黑窗)"""
    root = get_project_root()
    venv_pythonw = root / ".venv" / "Scripts" / "pythonw.exe"
    venv_python = root / ".venv" / "Scripts" / "python.exe"

    if venv_pythonw.exists():
        exe_path = venv_pythonw.resolve()
    elif venv_python.exists():
        exe_path = venv_python.resolve()
    else:
        exe_path = Path(sys.executable).resolve()

    run_tray_path = (root / "run_tray.py").resolve()
    return f'"{exe_path}" "{run_tray_path}" --minimized'


def is_autostart_enabled() -> bool:
    """檢查目前是否已啟用 Windows 登入自動啟動"""
    if not is_autostart_supported():
        return False

    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, REG_KEY_PATH, 0, winreg.KEY_READ
        ) as key:
            val, _ = winreg.QueryValueEx(key, APP_NAME)
            return bool(val)
    except FileNotFoundError:
        return False
    except Exception:
        return False


def set_autostart(enable: bool) -> Tuple[bool, str]:
    """設定或取消 Windows 登入自動啟動"""
    if not is_autostart_supported():
        return False, "目前作業系統非 Windows，不支援此登入自啟功能"

    import winreg

    cmd = get_autostart_command()

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, REG_KEY_PATH, 0, winreg.KEY_SET_VALUE
        ) as key:
            if enable:
                winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, cmd)
                return True, f"已成功設定 Windows 登入自動在背景啟動！\n命令: {cmd}"
            else:
                try:
                    winreg.DeleteValue(key, APP_NAME)
                except FileNotFoundError:
                    pass
                return True, "已取消 Windows 登入自動啟動"
    except Exception as e:
        return False, f"修改 Windows 註冊表失敗: {e}"


def get_autostart_status_info() -> dict:
    """取得詳細自啟狀態摘要"""
    enabled = is_autostart_enabled()
    cmd = get_autostart_command()
    return {
        "supported": is_autostart_supported(),
        "enabled": enabled,
        "command": cmd,
        "reg_path": f"HKCU\\{REG_KEY_PATH}\\{APP_NAME}",
    }
