"""
系統匣常駐模式整合測試 (scripts/test_tray.py)
驗證：
1. QSystemTrayIcon 與圖示載入/備援機制
2. 系統匣快顯選單項目完整性 (開啟/隱藏 Chat UI, Swagger 文件, 模型切換子選單, 顯存釋放, 偏好設定, 結束)
3. 視窗隱藏與喚醒邏輯 (show_chat_ui, hide_chat_ui, toggle_chat_ui)
4. 關閉事件攔截 (closeEvent -> 縮小至系統匣而不退出)
5. 健康檢查狀態同步與 Tooltip 動態更新
"""

import io
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 加入專案根目錄
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from src.ui.main_window import MainWindow
from src.ui.tray_manager import StrataTrayManager, get_app_icon


def test_tray_integration():
    print("=" * 65)
    print("   OpenVINO-Strata 系統匣 (System Tray) 模式整合測試")
    print("=" * 65)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    # 1. 測試圖示
    icon = get_app_icon()
    assert not icon.isNull(), "圖示載入失敗 (icon is null)"
    print("[✓] 1. 圖示載入測試通過")

    # 2. 測試主視窗與系統匣綁定
    window = MainWindow(base_url="http://127.0.0.1:12223", enable_tray=True)
    tray = window.tray_manager
    assert tray is not None, "MainWindow 系統匣管理器未成功建立"
    assert hasattr(tray, "tray_icon"), "tray_icon 物件不存在"
    print("[✓] 2. MainWindow 與 StrataTrayManager 綁定測試通過")

    # 3. 測試選單結構與關鍵項目
    actions = [a.text() for a in tray.menu.actions()]
    print("[*] 系統匣選單項目清單:")
    for a in actions:
        if a:
            print(f"    - {a}")

    assert any("開啟 Chat UI" in a for a in actions), "缺少『開啟 Chat UI』選單項"
    assert any("隱藏 Chat UI" in a for a in actions), "缺少『隱藏 Chat UI』選單項"
    assert any("API 文檔" in a for a in actions), "缺少『API 文檔』選單項"
    assert any("釋放顯存" in a for a in actions), "缺少『釋放顯存』選單項"
    assert any("伺服器日誌" in a for a in actions), "缺少『伺服器日誌』選單項"
    assert any("偏好設定" in a for a in actions), "缺少『偏好設定』選單項"
    assert any("最小化至系統匣" in a for a in actions), "缺少『最小化至系統匣』選單項"
    assert any("自動在背景啟動" in a for a in actions), "缺少『登入自動啟動』選單項"
    assert any("結束 OpenVINO-Strata" in a for a in actions), "缺少『結束』選單項"
    print("[✓] 3. 系統匣選單核心操作項完整度測試通過")

    # 4. 測試視窗隱藏與喚醒
    window.show()
    assert window.isVisible(), "視窗未能正常顯示"

    tray.hide_chat_ui()
    assert not window.isVisible(), "hide_chat_ui() 未能隱藏視窗"

    tray.show_chat_ui()
    assert window.isVisible(), "show_chat_ui() 未能喚醒視窗"

    tray.toggle_chat_ui()
    assert not window.isVisible(), "toggle_chat_ui() 未能切換隱藏"

    tray.toggle_chat_ui()
    assert window.isVisible(), "toggle_chat_ui() 未能切換顯示"
    print("[✓] 4. Chat UI 顯示 / 隱藏 / 切換喚醒測試通過")

    # 5. 測試 closeEvent 攔截 (縮小至系統匣)
    window.close()
    assert not window.isVisible(), "視窗關閉後應處於隱藏狀態"
    assert tray.tray_icon.isVisible(), "關閉視窗後系統匣圖示應繼續常駐"
    print("[✓] 5. 視窗關閉攔截與縮小至系統匣常駐測試通過")

    # 6. 測試健康資料更新與 Tooltip 動態同步
    mock_data = {
        "status": "healthy",
        "device": "GPU",
        "model": "Qwen3.8-35B-A3B-Distill-int4-ov",
        "loaded": True,
        "offload_ratio": 35,
        "gpu_memory": {"allocated_memory_mb": 12576.0},
        "system_memory": {"available_ram_gb": 16.5},
        "available_models": [
            {"id": "Qwen3.8-35B-A3B-Distill-int4-ov", "recommended_offload_ratio": 35},
            {"id": "Qwen3.6-35B-A3B-int4-ov", "recommended_offload_ratio": 40},
        ],
    }
    tray._on_health_updated(True, mock_data)
    tooltip = tray.tray_icon.toolTip()
    assert "Qwen3.8-35B-A3B-Distill-int4-ov" in tooltip, f"Tooltip 未包含模型名稱: {tooltip}"
    assert "12576 MB" in tooltip, f"Tooltip 未包含顯存資訊: {tooltip}"
    assert "35%" in tooltip, f"Tooltip 未包含 Offload 資訊: {tooltip}"
    print(f"[*] 動態更新後的系統匣提示字串 (Tooltip):\n{tooltip}")
    print("[✓] 6. 即時顯存資訊與模型狀態動態同步測試通過")

    # 7. 測試模型切換子選單動態填入
    submenu_actions = [a.text() for a in tray.model_submenu.actions()]
    print("[*] 模型切換子選單項目:")
    for sa in submenu_actions:
        print(f"    - {sa}")
    assert any("Qwen3.8-35B-A3B-Distill" in sa for sa in submenu_actions), "子選單未列出 Qwen3.8"
    assert any("Qwen3.6-35B-A3B" in sa for sa in submenu_actions), "子選單未列出 Qwen3.6"
    assert any("✔" in sa for sa in submenu_actions), "當前載入模型應標註勾選符號"
    print("[✓] 7. 模型熱切換子選單測試通過")

    # 8. 測試 Chat 介面伺服器日誌檢視器 (ServerLogDialog)
    from src.ui.server_log_dialog import ServerLogDialog
    window._on_open_server_log()
    log_dlg = window._log_dialog
    assert log_dlg is not None, "ServerLogDialog 未能成功初始化"
    assert hasattr(log_dlg, "log_text"), "log_text 文字區域未建立"
    log_dlg.search_input.setText("INFO")
    assert log_dlg.search_input.text() == "INFO", "日誌搜尋過濾功能異常"
    log_dlg.close()
    print("[✓] 8. Chat 介面伺服器日誌檢視器 (ServerLogDialog) 測試通過")

    # 9. 測試 Windows 登入自動啟動註冊機制 (autostart)
    from src.utils.autostart import (
        is_autostart_supported,
        is_autostart_enabled,
        set_autostart,
        get_autostart_command,
    )
    if is_autostart_supported():
        orig_state = is_autostart_enabled()
        ok_enable, msg1 = set_autostart(True)
        assert ok_enable and is_autostart_enabled(), f"自啟動啟用失敗: {msg1}"
        ok_disable, msg2 = set_autostart(False)
        assert ok_disable and not is_autostart_enabled(), f"自啟動解除失敗: {msg2}"
        if orig_state:
            set_autostart(True)
        print("[✓] 9. Windows 登入自啟動註冊表安裝與解除機制測試通過")

    # 10. 安全清理退出
    tray.quit_application()
    print("[✓] 10. 系統匣完整退出清理測試通過")
    print("=" * 65)
    print("🎉 所有測試均成功通過！Windows System Tray 操作模式運作健全！")
    print("=" * 65)


if __name__ == "__main__":
    test_tray_integration()
