"""Convert the generated magenta atlas into transparent, foot-aligned frames."""
from pathlib import Path
from PIL import Image
import numpy as np

assets = Path(__file__).resolve().parents[1] / "assets"
source = Image.open(assets / "actors_sheet.jpg").convert("RGBA")
xs = [20, 524, 1030, 1536, 2030]
ys = [40, 539, 1026, 1512, 2002]
for row in range(4):
    for col in range(3):
        im = source.crop((xs[col], ys[row], xs[col + 1], ys[row + 1]))
        a = np.array(im)
        r, g, b = [a[:, :, i].astype(float) for i in range(3)]
        mask = (r > g * 1.35 + 25) & (b > g * 1.25 + 20)
        a[mask, 3] = 0
        im = Image.fromarray(a)
        bounds = im.getbbox()
        im = im.crop(bounds)
        im.thumbnail((160, 220), Image.Resampling.LANCZOS)
        frame = Image.new("RGBA", (192, 240))
        frame.alpha_composite(im, ((192 - im.width) // 2, 232 - im.height))
        frame.save(assets / f"actor_{row}_{col}.png")
