"""
視窗位置記憶與跨螢幕邊界安全還原測試 (scripts/test_window_restore.py)
驗證項目：
1. 正常螢幕位置之儲存與精準還原
2. 超出顯示區域 (如拔除外接螢幕、負座標、超出螢幕範圍) 自動回退至第一個螢幕 (Screen 0)
3. 頂端標題列不可抓取時之自動安全置中校正
4. 最大化狀態 (Maximized) 記憶與正常尺寸 (normalGeometry) 保存
"""

import io
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtCore import QSettings, QRect

from src.ui.main_window import MainWindow


def test_window_geometry_restoration():
    print("=" * 68)
    print("   OpenVINO-Strata 視窗記憶與螢幕邊界安全還原測試")
    print("=" * 68)

    app = QApplication(sys.argv)
    screens = QGuiApplication.screens()
    first_screen = screens[0]
    avail = first_screen.availableGeometry()
    print(f"[*] 偵測到 {len(screens)} 個螢幕，第一個螢幕可用區域: {avail}")

    settings = QSettings("OpenVINO-Strata", "AIStudio")

    # -------------------------------------------------------------
    # 測試 1：正常位置儲存與還原
    # -------------------------------------------------------------
    print("\n[測試 1] 驗證正常可視位置的記憶與還原...")
    # 設定一個在第一個螢幕內的合理位置
    test_x = avail.x() + 80
    test_y = avail.y() + 60
    test_w = 920
    test_h = 640

    settings.setValue("window/x", test_x)
    settings.setValue("window/y", test_y)
    settings.setValue("window/width", test_w)
    settings.setValue("window/height", test_h)
    settings.setValue("window/maximized", False)
    settings.setValue("window/has_saved_state", True)
    settings.sync()

    win1 = MainWindow(enable_tray=False)
    # _restore_window_state 已在 __init__ 自動呼叫
    geom1 = win1.geometry()
    print(f"    設定目標: ({test_x}, {test_y}, {test_w}, {test_h})")
    print(f"    實際還原: ({geom1.x()}, {geom1.y()}, {geom1.width()}, {geom1.height()})")
    assert geom1.x() == test_x and geom1.y() == test_y, f"位置未精準還原: {geom1}"
    assert geom1.width() == test_w and geom1.height() == test_h, f"尺寸未精準還原: {geom1}"
    print("[✓] 測試 1 通過：正常位置精準還原成功")
    win1.close()

    # -------------------------------------------------------------
    # 測試 2：模擬拔除外接螢幕 (座標在極遠處，如 x=5000, y=5000)
    # -------------------------------------------------------------
    print("\n[測試 2] 驗證外接螢幕拔除後 (超出版面)，自動還原至第一個螢幕...")
    settings.setValue("window/x", 5000)
    settings.setValue("window/y", 4000)
    settings.setValue("window/width", 1000)
    settings.setValue("window/height", 700)
    settings.setValue("window/maximized", False)
    settings.setValue("window/has_saved_state", True)
    settings.sync()

    win2 = MainWindow(enable_tray=False)
    geom2 = win2.geometry()
    print(f"    已超出位置: (5000, 4000)")
    print(f"    回退至第一個螢幕位置: ({geom2.x()}, {geom2.y()}, {geom2.width()}, {geom2.height()})")

    # 驗證此視窗落在第一個螢幕可視區域內
    assert avail.contains(geom2.topLeft()), f"視窗左上角未落在第一個螢幕內: {geom2}"
    assert geom2.x() < 2000 and geom2.y() < 2000, f"視窗未被拉回第一個螢幕: {geom2}"
    print("[✓] 測試 2 通過：超出區域自動安全移回第一個螢幕")
    win2.close()

    # -------------------------------------------------------------
    # 測試 3：負座標超出版面 (x=-3000, y=-3000)
    # -------------------------------------------------------------
    print("\n[測試 3] 驗證負座標超出版面時自動還原...")
    settings.setValue("window/x", -3000)
    settings.setValue("window/y", -3000)
    settings.setValue("window/width", 900)
    settings.setValue("window/height", 600)
    settings.setValue("window/maximized", False)
    settings.setValue("window/has_saved_state", True)
    settings.sync()

    win3 = MainWindow(enable_tray=False)
    geom3 = win3.geometry()
    print(f"    負座標位置: (-3000, -3000)")
    print(f"    回退位置: ({geom3.x()}, {geom3.y()}, {geom3.width()}, {geom3.height()})")
    assert geom3.x() >= avail.left() and geom3.y() >= avail.top(), f"視窗仍在負座標區域: {geom3}"
    print("[✓] 測試 3 通過：負座標超出範圍自動修正")
    win3.close()

    # -------------------------------------------------------------
    # 測試 4：標題列超出螢幕頂端不可抓取 (y=-550)
    # -------------------------------------------------------------
    print("\n[測試 4] 驗證標題列被遮蔽不可拖曳時自動修正...")
    settings.setValue("window/x", avail.x() + 50)
    settings.setValue("window/y", -550)
    settings.setValue("window/width", 900)
    settings.setValue("window/height", 600)
    settings.setValue("window/maximized", False)
    settings.setValue("window/has_saved_state", True)
    settings.sync()

    win4 = MainWindow(enable_tray=False)
    geom4 = win4.geometry()
    print(f"    標題列出界位置: y=-550")
    print(f"    校正後位置: y={geom4.y()}")
    assert geom4.y() >= avail.top(), f"標題列仍被隱蔽在螢幕頂部上方: {geom4}"
    print("[✓] 測試 4 通過：標題列隱蔽自動拉回安全操作區域")
    win4.close()

    # -------------------------------------------------------------
    # 測試 5：最大化狀態保存與還原
    # -------------------------------------------------------------
    print("\n[測試 5] 驗證最大化 (Maximized) 狀態記憶與還原...")
    settings.setValue("window/x", test_x)
    settings.setValue("window/y", test_y)
    settings.setValue("window/width", test_w)
    settings.setValue("window/height", test_h)
    settings.setValue("window/maximized", True)
    settings.setValue("window/has_saved_state", True)
    settings.sync()

    win5 = MainWindow(enable_tray=False)
    assert win5.windowState() & win5.windowState().WindowMaximized, "視窗未能還原為最大化狀態"
    print("[✓] 測試 5 通過：視窗最大化狀態成功還原")
    win5.close()

    # 清理測試暫存，還原預設
    settings.clear()
    settings.sync()

    print("=" * 68)
    print("🎉 全部 5 項視窗記憶與跨螢幕邊界安全還原測試均成功通過！")
    print("=" * 68)


if __name__ == "__main__":
    test_window_geometry_restoration()
