from PIL import Image, ImageDraw
from pathlib import Path

outdir = Path(__file__).parent
outdir.mkdir(parents=True, exist_ok=True)

# Simple synthetic "1": a vertical white line
img1 = Image.new("L", (28, 28), color=0)
d1 = ImageDraw.Draw(img1)
d1.line((14, 4, 14, 24), fill=255, width=6)
img1.save(outdir / "one.png")

# Simple synthetic "0": white ring ellipse
img0 = Image.new("L", (28, 28), color=0)
d0 = ImageDraw.Draw(img0)
d0.ellipse((4, 4, 24, 24), outline=255, width=4)
img0.save(outdir / "zero.png")

print(f"Wrote: {outdir / 'one.png'} and {outdir / 'zero.png'}")

