"""
OpenVINO-Strata 跨執行模式 (原始碼模式 vs PyInstaller 打包執行檔) 路徑解析工具 (src/utils/paths.py)
負責：
1. 精確判斷當前是否處於 PyInstaller 打包環境 (sys.frozen)
2. 取得持久化資料根目錄 (models/, logs/ 等存放於 .exe 同層目錄)
3. 取得唯讀內嵌靜態資源目錄 (_MEIPASS/assets/)
"""

import sys
from pathlib import Path


def is_frozen() -> bool:
    """判斷當前是否運行於編譯/打包後的執行檔模式"""
    return getattr(sys, "frozen", False)


def get_app_root() -> Path:
    """
    取得應用程式主目錄 (用於放置 models/, logs/, 快取等持久化目錄)：
    - 打包模式：回傳 .exe 所在目錄
    - 原始碼模式：回傳專案根目錄
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent.parent


def get_bundle_dir() -> Path:
    """
    取得打包唯讀資源目錄：
    - 打包模式：回傳 PyInstaller 解壓縮之臨時目錄 (sys._MEIPASS)
    - 原始碼模式：回傳專案根目錄
    """
    if is_frozen() and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS).resolve()
    return get_app_root()


def get_asset_path(filename: str) -> Path:
    """
    取得 assets/ 目錄下的靜態資源路徑 (如 icon.ico, icon.png)：
    優先順序：_MEIPASS 內嵌資源 -> .exe 同層 assets/ -> 預設路徑
    """
    # 1. 優先檢查內嵌解壓資源
    p_bundle = get_bundle_dir() / "assets" / filename
    if p_bundle.exists():
        return p_bundle

    # 2. 檢查外部同層 assets 目錄
    p_root = get_app_root() / "assets" / filename
    if p_root.exists():
        return p_root

    return p_bundle
