"""Build NextMove archetype badges: source animal illustration -> square
canvas + colored '#Archetype' ribbon baked into the PNG.
"""
import os
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# name (as used in the app) -> (source file, ribbon color)
# Colors match ARCH_COLORS already defined in frontend/index_v5.html,
# Polyvalent gets a new teal to match the chameleon.
ARCHETYPES = {
    "Builder":    ("castor.png",   "#9333ea"),
    "Leader":     ("lion.png",     "#16a34a"),
    "Operator":   ("loutre.png",   "#dc3545"),
    "Expert":     ("hibou.png",    "#3b5bdb"),
    "Polyvalent": ("cameleon.png", "#0d9488"),
    "Explorer":   ("renard.png",   "#2563eb"),
}

SRC_DIR = os.path.join(os.path.dirname(__file__), "badge_sources")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "images")

CANVAS = 640          # square output canvas
TOP_PAD = 40           # space above the illustration
RIBBON_H = 90           # ribbon height
RIBBON_GAP = 18         # gap between illustration and ribbon


def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def build_badge(name, src_file, color_hex):
    src = Image.open(f"{SRC_DIR}/{src_file}").convert("RGBA")

    # Trim the near-white margin around the illustration so every badge
    # frames its animal consistently regardless of the source crop.
    bg = Image.new("RGBA", src.size, (255, 255, 255, 255))
    diff = Image.new("RGBA", src.size)
    gray = src.convert("L")
    bbox = gray.point(lambda p: 255 if p < 245 else 0).getbbox()
    if bbox:
        pad = 6
        l, t, r, b = bbox
        l = max(0, l - pad); t = max(0, t - pad)
        r = min(src.width, r + pad); b = min(src.height, b + pad)
        src = src.crop((l, t, r, b))

    # Fit the illustration into the top area of the canvas, centered.
    illus_area_h = CANVAS - TOP_PAD - RIBBON_GAP - RIBBON_H
    illus_area_w = CANVAS - 2 * 40
    scale = min(illus_area_w / src.width, illus_area_h / src.height)
    new_w, new_h = int(src.width * scale), int(src.height * scale)
    src_resized = src.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (CANVAS, CANVAS), (255, 255, 255, 255))
    paste_x = (CANVAS - new_w) // 2
    paste_y = TOP_PAD + (illus_area_h - new_h) // 2
    canvas.paste(src_resized, (paste_x, paste_y), src_resized)

    # Ribbon banner at the bottom third.
    draw = ImageDraw.Draw(canvas)
    ribbon_top = CANVAS - RIBBON_H
    color_rgb = hex_to_rgb(color_hex)
    draw.rectangle([0, ribbon_top, CANVAS, CANVAS], fill=color_rgb + (255,))

    # Subtle darker accent stripe on top of the ribbon for a "badge" feel.
    accent = tuple(max(0, c - 35) for c in color_rgb)
    draw.rectangle([0, ribbon_top, CANVAS, ribbon_top + 5], fill=accent + (255,))

    label = f"#{name}"
    font_size = 46
    font = ImageFont.truetype(FONT_PATH, font_size)
    tb = draw.textbbox((0, 0), label, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    while tw > CANVAS - 60 and font_size > 20:
        font_size -= 2
        font = ImageFont.truetype(FONT_PATH, font_size)
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]

    tx = (CANVAS - tw) / 2 - tb[0]
    ty = ribbon_top + (RIBBON_H - th) / 2 - tb[1]
    draw.text((tx, ty), label, font=font, fill=(255, 255, 255, 255))

    out_path = f"{OUT_DIR}/{name.lower()}.png"
    canvas.save(out_path)
    print(f"wrote {out_path}  ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    for name, (src_file, color_hex) in ARCHETYPES.items():
        build_badge(name, src_file, color_hex)
