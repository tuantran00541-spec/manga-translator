import json, cv2, numpy as np
data = json.load(open("/home/hatch/workspace/nano-vi/kiuyha-vs-paddle-ch337/paddle_boxes.json"))
tot = loose = 0
worst = []
for key in sorted(data, key=lambda k: int(k)):
    e = data[key]
    img = cv2.imread(e["original"], cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    _, ink = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(ink) > ink.size / 2:
        ink = 255 - ink
    _, labels, stats, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), connectivity=8)
    for b in e["boxes"]:
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        tot += 1
        m = 0.03 * min(bw, bh)
        xs, ys, xe, ye = [], [], [], []
        for (x, y, w, h, area) in (tuple(int(v) for v in r) for r in stats[1:]):
            if area < 12:
                continue
            if not (w <= bw + 2 * m and h <= bh + 2 * m and x >= x1 - m and y >= y1 - m and x + w <= x2 + m and y + h <= y2 + m):
                continue
            xs.append(x); ys.append(y); xe.append(x + w); ye.append(y + h)
        if not xs:
            continue
        tw, th = max(xe) - min(xs), max(ye) - min(ys)
        ratio = (tw * th) / (bw * bh)
        if ratio < 0.5:
            loose += 1
        worst.append((ratio, key, (x1, y1, x2, y2), (min(xs), min(ys), max(xe), max(ye))))
print(f"boxes: {tot}, tight-area <50% of box: {loose} ({100*loose/max(tot,1):.0f}%)")
worst.sort()
for r in worst[:8]:
    print(f"ratio {r[0]:.2f} slice {r[1]} box {r[2]} tight {r[3]}")
