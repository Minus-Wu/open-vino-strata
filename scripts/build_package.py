"""
OpenVINO-Strata 單一執行檔打包自動化腳本 (scripts/build_package.py)
負責：
1. 檢查打包前相依環境與靜態圖示 (assets/icon.ico)
2. 調用 PyInstaller 依據 open_vino_strata.spec 進行一鍵單一執行檔編譯
3. 編譯完成後自動驗證產物 (dist/OpenVINO-Strata.exe) 與執行冒煙測試 (--smoke-test)
"""

import argparse
import io
import os
import subprocess
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

root_dir = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenVINO-Strata PyInstaller 打包建置工具"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=True,
        help="建置前清理既有 build/ 與暫存快取 (預設啟用)",
    )
    parser.add_argument(
        "--smoke-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="打包完成後自動執行產出的 .exe 進行冒煙驗證 (預設啟用)",
    )
    return parser.parse_args()


def ensure_icons():
    """確認 assets/icon.ico 與 icon.png 存在，若不存在則自動生成"""
    ico_path = root_dir / "assets" / "icon.ico"
    png_path = root_dir / "assets" / "icon.png"
    if not ico_path.exists() or not png_path.exists():
        print("[*] 偵測到圖示檔缺失，自動調用 scripts/generate_icons.py 生成...")
        subprocess.check_call([sys.executable, str(root_dir / "scripts" / "generate_icons.py")])


def main() -> int:
    args = parse_args()

    print("=" * 72)
    print("   OpenVINO-Strata 單一執行檔 (One-File EXE) 自動化打包建置")
    print("=" * 72)
    print(f"[*] 專案目錄: {root_dir}")
    print(f"[*] Python 解譯器: {sys.executable}")

    ensure_icons()

    spec_path = root_dir / "open_vino_strata.spec"
    if not spec_path.exists():
        print(f"[ERROR] 找不到規格檔: {spec_path}")
        return 1

    dist_dir = root_dir / "dist"
    build_dir = root_dir / "build"

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(spec_path),
        "--noconfirm",
    ]
    if args.clean:
        cmd.append("--clean")

    print(f"[*] 執行 PyInstaller 打包指令:\n    {' '.join(cmd)}")
    start_time = time.time()

    ret = subprocess.call(cmd, cwd=str(root_dir))
    elapsed = time.time() - start_time

    if ret != 0:
        print(f"\n[ERROR] PyInstaller 建置失敗 (結束代碼: {ret})")
        return ret

    exe_path = dist_dir / "OpenVINO-Strata.exe"
    if not exe_path.exists():
        print(f"\n[ERROR] 建置完成但未在預期路徑找到執行檔: {exe_path}")
        return 1

    size_mb = exe_path.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 72)
    print(f"🎉 打包建置成功！耗時: {elapsed:.1f} 秒")
    print(f"[*] 產出執行檔: {exe_path.resolve()}")
    print(f"[*] 檔案大小: {size_mb:.2f} MB")
    print("=" * 72)

    if args.smoke_test:
        print("\n[*] 正在對產出的 OpenVINO-Strata.exe 執行冒煙測試 (--smoke-test)...")
        test_start = time.time()
        test_proc = subprocess.run(
            [str(exe_path), "--smoke-test"],
            cwd=str(root_dir),
            capture_output=True,
            text=True,
            timeout=30,
        )
        test_elapsed = time.time() - test_start
        if test_proc.returncode == 0:
            print(f"[✓] 執行檔冒煙測試通過！(耗時: {test_elapsed:.1f}s)")
        else:
            print(f"[!] 執行檔冒煙測試失敗 (代碼: {test_proc.returncode}):")
            print(test_proc.stdout)
            print(test_proc.stderr)
            return test_proc.returncode

    print("\n[使用提示]")
    print(f"  1. 雙擊執行: {exe_path.name} (以 Windows 系統匣模式常駐，點圖示開啟 Chat UI)")
    print(f"  2. 伺服器模式: .\\dist\\{exe_path.name} --server --port 12223")
    print(f"  3. 桌面聊天室: .\\dist\\{exe_path.name} --gui")
    print(f"  4. 開機自啟動: .\\dist\\{exe_path.name} --install-startup")
    print(f"  5. 請確保模型目錄 (models/) 與該 .exe 位於同一資料夾或透過 --model 指定。\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
