"""
FastAPI 主應用程式定義、生命週期管理與閒置顯存回收監控 (Idle Watchdog)
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.engine.device_manager import StrataConfig
from src.engine.ov_strata_engine import OVStrataEngine
from src.server.routes import router, set_engine

logger = logging.getLogger("ov_strata.app")


def create_app(
    model_path: Optional[Union[str, Path]] = None,
    device: str = "GPU",
    strata_config: Optional[StrataConfig] = None,
    lazy_load: bool = False,
) -> FastAPI:
    """工廠函數：建立已完成 OpenVINO-Strata 引擎初始化的 FastAPI 應用實例"""
    cfg = strata_config or StrataConfig(device=device.upper())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(
            f"Initializing OpenVINO-Strata engine on [{cfg.device}] "
            f"(model={model_path}, offload_ratio={cfg.offload_ratio}, vram_budget={cfg.vram_budget_gb}GB)..."
        )
        engine = OVStrataEngine(
            model_path=model_path,
            device=cfg.device,
            strata_config=cfg,
            auto_load=not lazy_load and (model_path is not None),
        )
        set_engine(engine)

        # 若設定了閒置自動卸載秒數 (idle_unload_seconds > 0)，啟動背景看門狗
        watchdog_task: Optional[asyncio.Task] = None

        async def _idle_watchdog():
            while True:
                await asyncio.sleep(15)
                idle_limit = engine.strata_config.idle_unload_seconds
                if idle_limit > 0 and engine.is_loaded:
                    elapsed = time.time() - engine._last_active_time
                    if elapsed >= idle_limit:
                        logger.info(
                            f"Model [{engine.model_name}] idle for {elapsed:.0f}s (>= {idle_limit}s); "
                            "auto-unloading from VRAM..."
                        )
                        await engine.unload_model()

        watchdog_task = asyncio.create_task(_idle_watchdog())
        logger.info("OpenVINO-Strata LLM Serving ready to accept requests.")
        yield

        if watchdog_task:
            watchdog_task.cancel()
        logger.info("Shutting down OpenVINO-Strata LLM Serving and releasing VRAM...")
        await engine.unload_model()

    app = FastAPI(
        title="OpenVINO-Strata LLM Serving",
        description=(
            "Low-VRAM MoE LLM Provider powered by Intel OpenVINO 2026 & PolyStrata Tiered Memory Architecture "
            "(OpenAI & Anthropic API compatible)."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app
