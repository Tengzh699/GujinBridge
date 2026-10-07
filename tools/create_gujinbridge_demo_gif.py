#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render a small README demo GIF from verified GujinBridge DPO outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CASES = [
    (
        "01  古文译现代汉语",
        "臣本布衣，躬耕于南阳，苟全性命于乱世，不求闻达于诸侯。",
        "我本来是平民百姓，在南阳亲自耕种，只希望在乱世中保全性命，不求在诸侯间出名显贵。",
        "2.92s",
    ),
    (
        "02  现代汉语译古文",
        "学习贵在长期坚持，即使遇到困难，也不应该半途而废。",
        "学贵恒心，虽有难而不废也。",
        "1.03s",
    ),
    (
        "03  古文断句与标点",
        "故天将降大任于是人也必先苦其心志劳其筋骨饿其体肤",
        "故天将降大任于是人也，必先苦其心志，劳其筋骨，饿其体肤。",
        "2.27s",
    ),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path(r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def wrap(draw: ImageDraw.ImageDraw, text: str, selected_font, width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for char in text:
        proposed = current + char
        if current and draw.textbbox((0, 0), proposed, font=selected_font)[2] > width:
            lines.append(current)
            current = char
        else:
            current = proposed
    if current:
        lines.append(current)
    return lines


def render(visible_cases: int) -> Image.Image:
    image = Image.new("RGB", (1200, 760), "#0b1020")
    draw = ImageDraw.Draw(image)
    title_font = font(34, bold=True)
    subtitle_font = font(17)
    label_font = font(20, bold=True)
    body_font = font(20)
    small_font = font(15)

    draw.rounded_rectangle((38, 30, 1162, 730), radius=24, fill="#111827", outline="#334155", width=2)
    draw.ellipse((70, 57, 86, 73), fill="#fb7185")
    draw.ellipse((96, 57, 112, 73), fill="#fbbf24")
    draw.ellipse((122, 57, 138, 73), fill="#34d399")
    draw.text((170, 48), "GujinBridge · DPO V1", font=title_font, fill="#f8fafc")
    draw.text((825, 56), "Qwen3-1.7B  ·  RTX 3090", font=subtitle_font, fill="#94a3b8")
    draw.line((70, 105, 1130, 105), fill="#334155", width=2)

    top = 130
    for index, (label, prompt, answer, elapsed) in enumerate(CASES):
        active = index < visible_cases
        panel_top = top + index * 188
        panel_bottom = panel_top + 164
        fill = "#172033" if active else "#131a2a"
        outline = "#3b82f6" if active and index == visible_cases - 1 else "#263348"
        draw.rounded_rectangle((70, panel_top, 1130, panel_bottom), radius=16, fill=fill, outline=outline, width=2)
        if not active:
            draw.text((96, panel_top + 64), label, font=label_font, fill="#475569")
            continue

        draw.text((96, panel_top + 18), label, font=label_font, fill="#60a5fa")
        draw.text((1016, panel_top + 21), elapsed, font=small_font, fill="#64748b")
        draw.text((96, panel_top + 57), "输入", font=small_font, fill="#94a3b8")
        prompt_lines = wrap(draw, prompt, body_font, 900)
        draw.text((150, panel_top + 50), "\n".join(prompt_lines), font=body_font, fill="#cbd5e1", spacing=5)
        answer_y = panel_top + 91 if len(prompt_lines) == 1 else panel_top + 116
        draw.text((96, answer_y + 7), "输出", font=small_font, fill="#94a3b8")
        answer_lines = wrap(draw, answer, body_font, 900)
        draw.text((150, answer_y), "\n".join(answer_lines), font=body_font, fill="#f8fafc", spacing=5)

    draw.text((72, 702), "真实本地推理输出 · 贪心解码 · 禁用 thinking", font=small_font, fill="#64748b")
    draw.text((950, 702), "GujinBridge", font=small_font, fill="#60a5fa")
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the GujinBridge README demo GIF")
    parser.add_argument("--output", type=Path, default=Path("docs/assets/gujinbridge-demo.gif"))
    parser.add_argument("--png", type=Path, default=Path("docs/assets/gujinbridge-demo.png"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.png.parent.mkdir(parents=True, exist_ok=True)
    frames = [render(count) for count in (0, 1, 2, 3)]
    frames[-1].save(args.png, optimize=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=[700, 1800, 1800, 4200],
        loop=0,
        optimize=True,
    )
    print(f"GIF: {args.output.resolve()}")
    print(f"PNG: {args.png.resolve()}")


if __name__ == "__main__":
    main()
