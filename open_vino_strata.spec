# -*- mode: python ; coding: utf-8 -*-
"""
OpenVINO-Strata PyInstaller 單一執行檔打包規格檔 (open_vino_strata.spec)
將 OpenVINO 2026 GenAI、PyQt6 桌面聊天室、FastAPI 伺服器與系統匣管理模組
完整封裝為獨立的單一執行檔 (OpenVINO-Strata.exe)。
"""

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files, collect_submodules

block_cipher = None
spec_root = Path(SPECPATH).resolve()

# 1. 收集動態函式庫 (OpenVINO C++ 插件, GenAI, Tokenizers, TBB 等 25+ 個核心 DLL)
binaries = []
binaries += collect_dynamic_libs('openvino')
binaries += collect_dynamic_libs('openvino_genai')
binaries += collect_dynamic_libs('openvino_tokenizers')

# 2. 收集靜態資料檔案 (assets 圖示、OpenVINO 核心設定)
datas = [
    (str(spec_root / 'assets'), 'assets'),
]
datas += collect_data_files('openvino')
datas += collect_data_files('openvino_genai')
datas += collect_data_files('openvino_tokenizers')

# 3. 隱含匯入清單 (Uvicorn 協定、FastAPI 路由、PyQt6 與本專案模組)
hiddenimports = [
    # OpenVINO 核心
    'openvino',
    'openvino._pyopenvino',
    'openvino.runtime',
    'openvino.properties',
    'openvino.package_utils',
    'openvino_genai',
    'openvino_genai.py_openvino_genai',
    'openvino_tokenizers',

    # Uvicorn & FastAPI 服務層
    'uvicorn',
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.http.h11_impl',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'fastapi',
    'fastapi.responses',
    'starlette',
    'starlette.responses',
    'pydantic',
    'pydantic.deprecated',
    'pydantic.deprecated.decorator',
    'pydantic_core',
    'httpx',
    'psutil',
    'PIL',
    'PIL.Image',

    # PyQt6 UI 視窗元件
    'PyQt6',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',

    # 本專案內部模組與多模式進入點
    'run_server',
    'run_gui',
    'src.engine.ov_strata_engine',
    'src.engine.device_manager',
    'src.engine.parser',
    'src.server.app',
    'src.server.routes',
    'src.server.schemas',
    'src.ui.main_window',
    'src.ui.chat_widget',
    'src.ui.api_worker',
    'src.ui.settings_dialog',
    'src.ui.server_log_dialog',
    'src.ui.tray_manager',
    'src.ui.styles',
    'src.utils.autostart',
    'src.utils.paths',
]

# 4. 排除非推論必要之巨型訓練/轉換庫 (如 PyTorch, Transformers, TensorFlow) 以確保檔案精簡與極速編譯
excludes = [
    'torch',
    'torchvision',
    'torchaudio',
    'transformers',
    'optimum',
    'nncf',
    'accelerate',
    'tensorflow',
    'tensorboard',
    'scipy',
    'sklearn',
    'scikit-learn',
    'matplotlib',
    'tkinter',
    'pytest',
    'unittest',
    'notebook',
    'IPython',
]

a = Analysis(
    ['run_tray.py'],
    pathex=[str(spec_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='OpenVINO-Strata',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,              # UPX 可能損壞 C++ runtime 與 Qt DLL，設為 False
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # 設為 False 避免啟動時彈出黑色命令提示字元黑窗
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(spec_root / 'assets' / 'icon.ico'),
)
