"""
產生 OpenVINO-Strata 高解析度與多尺寸圖示 (assets/icon.png, assets/icon.ico)
設計理念：
- 科技感深色微光圓角六角形外框
- Intel Arc / OpenVINO 科技藍與霓虹青色漸層
- PolyStrata 分層階梯 (Stratified Layers) 與中央晶片運算核心
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter


def create_strata_icon(size: int = 256) -> Image.Image:
    # 建立具有透明通道的高解析度畫布
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    pad = size * 0.08
    r = size * 0.22

    # 1. 繪製圓角矩形背景 (深色金屬感微光漸層)
    # 底層陰影微光
    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.rounded_rectangle(
        [pad - 2, pad - 2, size - pad + 2, size - pad + 2],
        radius=int(r),
        fill=(14, 165, 233, 110),
    )
    glow = glow.filter(ImageFilter.GaussianBlur(radius=size * 0.04))
    img = Image.alpha_composite(glow, img)
    draw = ImageDraw.Draw(img)

    # 主體底板
    draw.rounded_rectangle(
        [pad, pad, size - pad, size - pad],
        radius=int(r),
        fill=(15, 23, 42, 245),
        outline=(56, 189, 248, 220),
        width=max(2, int(size * 0.02)),
    )

    # 2. 繪製 PolyStrata 分層橫條 (三層動態分層：L1 GPU VRAM / L2 Shared / L3 System RAM)
    cx = size / 2.0
    cy = size / 2.0
    layer_w = size * 0.56
    layer_h = size * 0.09
    spacing = size * 0.06

    layers = [
        # (Y 偏移, 寬度比例, 填色, 外框)
        (cy - spacing * 1.8, 0.75, (14, 165, 233, 240), (125, 211, 252, 255)),  # Top: 高速 GPU (Cyan)
        (cy - spacing * 0.2, 0.95, (59, 130, 246, 240), (147, 197, 253, 255)),  # Mid: 共享熱點 (Blue)
        (cy + spacing * 1.4, 0.82, (139, 92, 246, 240), (196, 181, 253, 255)),  # Bottom: 系統 RAM (Purple)
    ]

    for y_pos, w_scale, fill_color, border_color in layers:
        w = layer_w * w_scale
        x0 = cx - w / 2.0
        x1 = cx + w / 2.0
        y0 = y_pos - layer_h / 2.0
        y1 = y_pos + layer_h / 2.0
        draw.rounded_rectangle(
            [x0, y0, x1, y1],
            radius=int(layer_h * 0.45),
            fill=fill_color,
            outline=border_color,
            width=max(1, int(size * 0.012)),
        )

    # 3. 繪製中央微光核心 (Intel 晶片風格加速節點)
    core_r = size * 0.075
    draw.ellipse(
        [cx - core_r, cy - core_r, cx + core_r, cy + core_r],
        fill=(255, 255, 255, 250),
        outline=(56, 189, 248, 255),
        width=max(1, int(size * 0.015)),
    )

    # 4. 微光小閃電 / 加速箭頭
    spark_points = [
        (cx - size * 0.02, cy - size * 0.04),
        (cx + size * 0.025, cy - size * 0.01),
        (cx - size * 0.005, cy + size * 0.005),
        (cx + size * 0.02, cy + size * 0.04),
        (cx - size * 0.025, cy + size * 0.01),
        (cx + size * 0.005, cy - size * 0.005),
    ]
    draw.polygon(spark_points, fill=(2, 132, 199, 255))

    return img


def main():
    root = Path(__file__).resolve().parent.parent
    assets_dir = root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    png_path = assets_dir / "icon.png"
    ico_path = assets_dir / "icon.ico"

    base_img = create_strata_icon(512)
    # 存檔 PNG (256x256)
    img_256 = base_img.resize((256, 256), Image.Resampling.LANCZOS)
    img_256.save(png_path, "PNG")
    print(f"[OK] 已產生 PNG 圖示: {png_path}")

    # 存檔多尺寸 ICO (16, 24, 32, 48, 64, 128, 256)
    sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    base_img.save(ico_path, format="ICO", sizes=sizes)
    print(f"[OK] 已產生 Windows 多尺寸 ICO 圖示: {ico_path}")


if __name__ == "__main__":
    main()
