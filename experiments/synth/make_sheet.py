"""Build visual comparison sheets: line render | toon | generated variants per scene.

  genv/bin/python make_sheet.py --gen r2a/gen --renders r2a/renders_v2 \
      --out r2a/sheets --per-sheet 4
"""
import argparse
import os

from PIL import Image, ImageDraw

THUMB_H = 512


def load(fp):
    try:
        im = Image.open(fp).convert("RGB")
        return im.resize((int(im.width * THUMB_H / im.height), THUMB_H))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="r2a/gen")
    ap.add_argument("--renders", default="r2a/renders_v2")
    ap.add_argument("--out", default="r2a/sheets")
    ap.add_argument("--per-sheet", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    scenes = sorted(s for s in os.listdir(args.gen)
                    if os.path.isdir(os.path.join(args.gen, s)))
    rows = []
    for scene in scenes:
        cells = [("line(3D)", load(os.path.join(args.renders, scene, "line.png"))),
                 ("toon(3D)", load(os.path.join(args.renders, scene, "toon.png")))]
        gen_files = sorted(f for f in os.listdir(os.path.join(args.gen, scene))
                           if f.endswith(".png"))
        for f in gen_files:
            cells.append((f.replace(".png", ""), load(os.path.join(args.gen, scene, f))))
        cells = [(t, im) for t, im in cells if im is not None]
        if len(cells) > 2:
            rows.append((scene, cells))

    for si in range(0, len(rows), args.per_sheet):
        chunk = rows[si:si + args.per_sheet]
        w = max(sum(im.width + 8 for _, im in cells) for _, cells in chunk) + 8
        h = len(chunk) * (THUMB_H + 40) + 8
        sheet = Image.new("RGB", (w, h), (245, 245, 245))
        dr = ImageDraw.Draw(sheet)
        y = 8
        for scene, cells in chunk:
            x = 8
            for title, im in cells:
                sheet.paste(im, (x, y + 32))
                dr.text((x, y + 16), title, fill=(0, 0, 0))
                x += im.width + 8
            dr.text((8, y), scene, fill=(180, 0, 0))
            y += THUMB_H + 40
        fp = os.path.join(args.out, f"sheet_{si // args.per_sheet:02d}.png")
        sheet.save(fp)
        print("wrote", fp, f"({len(chunk)} scenes)")


if __name__ == "__main__":
    main()
