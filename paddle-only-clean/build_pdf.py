import re, os
from PIL import Image, ImageDraw, ImageFont

OUT = "out"
ORIG = "/home/hatch/workspace/fix-audit-h1-wt/data/raw/a0000162/sliced"
PDF = "paddle-clean-ch337-review.pdf"
W = 640

# box counts + times from the run log
meta = {}
for line in open("full_run.log"):
    m = re.match(r"\[(\d+)\] slice \d+: (\d+) boxes, ([\d.]+)s -> .*/out/(\S+)", line)
    if m:
        meta[m.group(4)] = (int(m.group(1)), int(m.group(2)), float(m.group(3)))

names = sorted(os.listdir(OUT))
print("slices:", len(names), "with meta:", len(meta))

def fit(im):
    if im.width > W:
        im = im.resize((W, int(im.height * W / im.width)), Image.LANCZOS)
    return im.convert("RGB")

def strip(text, h=44, bg=(20, 20, 20), fg=(255, 255, 255), size=26):
    img = Image.new("RGB", (W, h), bg)
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        f = ImageFont.load_default()
    d.text((12, (h - size) // 2 - 2), text, font=f, fill=fg)
    return img

pages = []
for name in names:
    idx, boxes, secs = meta.get(name, (-1, -1, -1))
    orig = fit(Image.open(os.path.join(ORIG, name)))
    clean = fit(Image.open(os.path.join(OUT, name)))
    header = strip(f"slice {idx}  {name}  -  {boxes} boxes  -  {secs:.1f}s")
    divider = strip("DA CLEAN  (duoi)  |  GOC  (tren)", h=30, bg=(70, 70, 70), size=20)
    page = Image.new("RGB", (W, header.height + orig.height + divider.height + clean.height), (255, 255, 255))
    y = 0
    for part in (header, orig, divider, clean):
        page.paste(part, (0, y)); y += part.height
    pages.append(page)
    if len(pages) % 20 == 0:
        print("composed", len(pages))

pages[0].save(PDF, "PDF", save_all=True, append_images=pages[1:], resolution=100.0)
print("saved", PDF, os.path.getsize(PDF) / 1e6, "MB, pages:", len(pages))
