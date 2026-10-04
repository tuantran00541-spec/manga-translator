"""Hand-label text on processed slices and score the app's text detection and erasing against those labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from app.config import BASE_DIR, PROCESSED_DIR
from app.manifest_utils import load_manifest_raw

KINDS = ("text", "sfx", "watermark")
VERDICTS = ("good", "text_left", "bad_fill")
FOUND_COVER = 0.5  # share of a labelled box the app's boxes must cover to count as found
FALSE_OVERLAP = 0.1  # app boxes overlapping labels less than this are false finds
_lock = threading.Lock()


def _path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


def _sha1(path: Path | None) -> str | None:
    return hashlib.sha1(path.read_bytes()).hexdigest() if path and path.is_file() else None


def _page(chapter: str, number: int) -> dict:
    return load_manifest_raw(chapter)["pages"][number]


def _size(path: Path) -> tuple[int, int]:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return int(image.shape[1]), int(image.shape[0])


def list_chapters() -> list[dict]:
    found = []
    for manifest in sorted(PROCESSED_DIR.glob("*/manifest.json")):
        data = load_manifest_raw(manifest.parent.name)
        pages = [p for p in data.get("pages", []) if not p.get("skipped") and p.get("clean")]
        found.append({"id": manifest.parent.name, "source": data.get("source_url") or data.get("title") or "",
                      "cleaned": len(pages)})
    return found


def sample(labels: dict, chapters: list[str], count: int, seed: int) -> None:
    """Add a random spread of cleaned slices from new chapters to the label set."""
    have = {(s["chapter"], s["page"]) for s in labels["slices"]}
    pool = []
    for chapter in chapters:
        for number, page in enumerate(load_manifest_raw(chapter)["pages"]):
            if not page.get("skipped") and page.get("clean") and (chapter, number) not in have:
                pool.append((chapter, number))
    picked = sorted(random.Random(seed).sample(pool, min(count, len(pool))))
    for chapter, number in picked:
        original = _path(_page(chapter, number)["original"])
        width, height = _size(original)
        labels["slices"].append({"chapter": chapter, "page": number, "w": width, "h": height,
                                 "original_sha1": _sha1(original), "done": False, "boxes": [], "verdicts": {}})


def _app_boxes(page: dict, width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    for box in page.get("boxes") or []:
        if box.get("removed") or box.get("manual") or box.get("origin") == "manual":
            continue
        x1, y1 = max(0, int(box["x1"])), max(0, int(box["y1"]))
        x2, y2 = min(width, int(box["x2"])), min(height, int(box["y2"]))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def score(labels: dict) -> dict:
    """Found, false and erase-quality counts over the slices marked done."""
    by_kind = {k: {"labels": 0, "found": 0, "good": 0, "text_left": 0, "bad_fill": 0, "unjudged": 0} for k in KINDS}
    result = {"slices_done": 0, "slices_total": len(labels["slices"]), "slices_changed": 0,
              "app_boxes": 0, "false_boxes": 0, "missed": [], "by_kind": by_kind}
    for index, item in enumerate(labels["slices"]):
        if not item["done"]:
            continue
        page = _page(item["chapter"], item["page"])
        if _sha1(_path(page["original"])) != item["original_sha1"]:
            result["slices_changed"] += 1
            continue
        result["slices_done"] += 1
        width, height = item["w"], item["h"]
        app_mask = np.zeros((height, width), bool)
        label_mask = np.zeros((height, width), bool)
        app_boxes = _app_boxes(page, width, height)
        for x1, y1, x2, y2 in app_boxes:
            app_mask[y1:y2, x1:x2] = True
        for box in item["boxes"]:
            label_mask[box["y1"]:box["y2"], box["x1"]:box["x2"]] = True
        clean_sha1 = _sha1(_path(page.get("clean")))
        for box in item["boxes"]:
            tally = by_kind[box["kind"]]
            tally["labels"] += 1
            area = app_mask[box["y1"]:box["y2"], box["x1"]:box["x2"]]
            if area.size and area.mean() >= FOUND_COVER:
                tally["found"] += 1
            else:
                result["missed"].append({"slice": index, "box": box["id"], "kind": box["kind"]})
            verdict = item["verdicts"].get(box["id"])
            if verdict and verdict.get("clean_sha1") == clean_sha1:
                tally[verdict["verdict"]] += 1
            else:
                tally["unjudged"] += 1
        for x1, y1, x2, y2 in app_boxes:
            result["app_boxes"] += 1
            if label_mask[y1:y2, x1:x2].mean() < FALSE_OVERLAP:
                result["false_boxes"] += 1
    for tally in by_kind.values():
        judged = tally["good"] + tally["text_left"] + tally["bad_fill"]
        tally["found_rate"] = round(tally["found"] / tally["labels"], 3) if tally["labels"] else None
        tally["good_rate"] = round(tally["good"] / judged, 3) if judged else None
    return result


def _save(path: Path, labels: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def _clean_box(raw: dict, width: int, height: int) -> dict | None:
    x1, x2 = sorted((max(0, min(width, int(raw["x1"]))), max(0, min(width, int(raw["x2"])))))
    y1, y2 = sorted((max(0, min(height, int(raw["y1"]))), max(0, min(height, int(raw["y2"])))))
    if x2 - x1 < 3 or y2 - y1 < 3 or raw.get("kind") not in KINDS:
        return None
    return {"id": str(raw["id"])[:40], "x1": x1, "y1": y1, "x2": x2, "y2": y2, "kind": raw["kind"]}


def make_handler(labels_path: Path, labels: dict):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, status: int, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, value, status: int = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(value, ensure_ascii=False).encode(), "application/json")

        def do_GET(self):
            parts = self.path.split("?")[0].strip("/").split("/")
            if parts == [""]:
                self._send(HTTPStatus.OK, PAGE.encode(), "text/html; charset=utf-8")
            elif parts == ["api", "state"]:
                with _lock:
                    self._json(labels)
            elif parts == ["api", "score"]:
                with _lock:
                    self._json(score(labels))
            elif len(parts) == 3 and parts[0] == "img" and parts[1].isdigit() and parts[2] in ("original", "clean"):
                index = int(parts[1])
                if index >= len(labels["slices"]):
                    return self._json({"error": "no slice"}, HTTPStatus.NOT_FOUND)
                item = labels["slices"][index]
                path = _path(_page(item["chapter"], item["page"]).get(parts[2]))
                if not path or not path.is_file():
                    return self._json({"error": "no image"}, HTTPStatus.NOT_FOUND)
                kind = {".png": "image/png", ".webp": "image/webp"}.get(path.suffix.lower(), "image/jpeg")
                self._send(HTTPStatus.OK, path.read_bytes(), kind)
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            parts = self.path.strip("/").split("/")
            if len(parts) != 3 or parts[:2] != ["api", "slice"] or not parts[2].isdigit():
                return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            index = int(parts[2])
            length = int(self.headers.get("Content-Length") or 0)
            if index >= len(labels["slices"]) or length > 1_000_000:
                return self._json({"error": "bad request"}, HTTPStatus.BAD_REQUEST)
            body = json.loads(self.rfile.read(length) or b"{}")
            with _lock:
                item = labels["slices"][index]
                if "boxes" in body:
                    boxes = [_clean_box(b, item["w"], item["h"]) for b in body["boxes"]]
                    item["boxes"] = [b for b in boxes if b]
                    keep = {b["id"] for b in item["boxes"]}
                    item["verdicts"] = {k: v for k, v in item["verdicts"].items() if k in keep}
                if "done" in body:
                    item["done"] = bool(body["done"])
                if body.get("verdict") and body["verdict"].get("verdict") in VERDICTS:
                    page = _page(item["chapter"], item["page"])
                    item["verdicts"][str(body["verdict"]["box"])] = {
                        "verdict": body["verdict"]["verdict"], "clean_sha1": _sha1(_path(page.get("clean")))}
                _save(labels_path, labels)
                self._json(item)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=BASE_DIR / "data" / "labels" / "labels.json")
    parser.add_argument("--chapters", nargs="*", default=[], help="processed chapter ids to sample slices from")
    parser.add_argument("--slices", type=int, default=60, help="slices to sample over the new chapters")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--score", action="store_true", help="print the score and exit")
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to label from a phone on the same wifi")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    labels = json.loads(args.labels.read_text(encoding="utf-8")) if args.labels.is_file() else {"slices": []}
    if args.chapters:
        sample(labels, args.chapters, args.slices, args.seed)
        _save(args.labels, labels)
    if args.score:
        print(json.dumps(score(labels), ensure_ascii=False, indent=1))
        return 0
    if not labels["slices"]:
        print("No slices yet. Processed chapters (pass ids with --chapters):")
        for chapter in list_chapters():
            print(f"  {chapter['id']}  {chapter['cleaned']} cleaned slices  {chapter['source']}")
        return 1
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.labels, labels))
    print(f"Labelling {len(labels['slices'])} slices at http://{args.host}:{args.port}  (labels: {args.labels})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


PAGE = r"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Đánh dấu chữ</title><link rel="icon" href="data:,">
<style>
:root{--bg:#f6f6f4;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#d9d9d4;--accent:#2563eb;
--text:#e11d48;--sfx:#d97706;--watermark:#7c3aed}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecece8;--muted:#9a9a94;--card:#222220;--line:#3a3a37}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:8px 16px;
display:flex;flex-wrap:wrap;gap:8px;align-items:center}
button{font:inherit;padding:7px 12px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);cursor:pointer}
button.on{background:var(--accent);border-color:var(--accent);color:#fff}
button.k-text.on{background:var(--text);border-color:var(--text)}button.k-sfx.on{background:var(--sfx);border-color:var(--sfx)}
button.k-watermark.on{background:var(--watermark);border-color:var(--watermark)}
.grow{flex:1}.muted{color:var(--muted);font-size:13px}main{padding:12px 16px;max-width:1100px;margin:0 auto}
#wrap{position:relative;width:100%}#wrap canvas{width:100%;display:block;border-radius:6px}
#wrap.draw canvas{touch-action:none;cursor:crosshair}
.pair{display:flex;gap:8px}.pair figure{flex:1;margin:0}.pair canvas{width:100%;background:#000;border-radius:6px}
.pair figcaption{font-size:13px;color:var(--muted)}
table{border-collapse:collapse;width:100%;background:var(--card)}td,th{border:1px solid var(--line);padding:6px 8px;text-align:left}
.help{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin-bottom:12px;font-size:14px}
</style></head><body>
<header>
 <button data-mode="mark" class="on">Đánh dấu</button><button data-mode="judge">Chấm xóa</button><button data-mode="score">Kết quả</button>
 <span class="grow"></span><span id="progress" class="muted"></span>
</header>
<main>
 <section id="mark">
  <div class="help">Khoanh <b>mọi chữ</b> trên ảnh gốc, mỗi khung một cụm chữ (bong bóng, ô tường thuật, chữ ngoài khung).
  Chọn loại trước khi vẽ: <b>Chữ</b> = thoại/tường thuật cần xóa, <b>SFX</b> = chữ hiệu ứng vẽ tay, <b>Watermark</b> = logo nhóm dịch.
  Bấm vào khung để chọn, đổi loại hoặc xóa. Lát không có chữ thì bấm luôn “Xong”.
  <span class="muted">Phím: 1/2/3 loại · Del xóa · Enter xong và sang lát sau · A/D lát trước/sau.</span></div>
  <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px;align-items:center">
   <button class="k-text on" data-kind="text">1 Chữ</button><button class="k-sfx" data-kind="sfx">2 SFX</button>
   <button class="k-watermark" data-kind="watermark">3 Watermark</button>
   <button id="del">Xóa khung</button><button id="scroll">Đang vẽ (chạm để cuộn)</button>
   <span class="grow"></span><button id="prev">← Lát trước</button><button id="done" class="on">Xong → lát sau</button>
  </div>
  <div id="sliceinfo" class="muted" style="margin-bottom:6px"></div>
  <div id="wrap" class="draw"><canvas id="cv"></canvas></div>
 </section>
 <section id="judge" hidden>
  <div class="help">So từng khung đã đánh dấu: trái là ảnh gốc, phải là ảnh app đã xóa chữ.
  <b>Sạch đẹp</b> = hết chữ và nền trông tự nhiên · <b>Còn chữ</b> = còn nét hoặc bóng chữ · <b>Nền xấu</b> = hết chữ nhưng nền nhòe, lệch màu, mất nét viền.
  <span class="muted">Phím: 1/2/3, S bỏ qua.</span></div>
  <div id="jinfo" class="muted" style="margin-bottom:6px"></div>
  <div class="pair"><figure><figcaption>Gốc</figcaption><canvas id="jo"></canvas></figure>
   <figure><figcaption>Sau khi xóa</figcaption><canvas id="jc"></canvas></figure></div>
  <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:10px">
   <button data-verdict="good">1 Sạch đẹp</button><button data-verdict="text_left">2 Còn chữ</button>
   <button data-verdict="bad_fill">3 Nền xấu</button><button id="skip">S Bỏ qua</button></div>
 </section>
 <section id="score" hidden><div id="scorebox"></div></section>
</main>
<script>
const COLORS={text:'#e11d48',sfx:'#d97706',watermark:'#7c3aed'};
const NAMES={text:'Chữ',sfx:'SFX',watermark:'Watermark'};
let state=null,cur=0,kind='text',sel=null,img=new Image(),drag=null,drawMode=true,mode='mark';
const cv=document.getElementById('cv'),ctx=cv.getContext('2d'),wrap=document.getElementById('wrap');
const $=id=>document.getElementById(id);
async function load(){state=await (await fetch('/api/state')).json();
 cur=Math.max(0,state.slices.findIndex(s=>!s.done));if(cur<0)cur=0;showSlice();}
function progress(){const d=state.slices.filter(s=>s.done).length;
 $('progress').textContent=`Đã xong ${d}/${state.slices.length} lát`;}
function item(){return state.slices[cur];}
function showSlice(){const it=item();sel=null;progress();
 $('sliceinfo').textContent=`Lát ${cur+1}/${state.slices.length} · chương ${it.chapter} trang ${it.page}${it.done?' · đã xong':''}`;
 img=new Image();img.onload=()=>{cv.width=img.naturalWidth;cv.height=img.naturalHeight;draw();};
 img.src=`/img/${cur}/original`;window.scrollTo(0,0);}
function draw(){ctx.drawImage(img,0,0);const lw=Math.max(2,cv.width/300);
 for(const b of item().boxes){ctx.lineWidth=b===sel?lw*2:lw;ctx.strokeStyle=COLORS[b.kind];
  ctx.fillStyle=COLORS[b.kind]+(b===sel?'44':'22');ctx.fillRect(b.x1,b.y1,b.x2-b.x1,b.y2-b.y1);
  ctx.strokeRect(b.x1,b.y1,b.x2-b.x1,b.y2-b.y1);}
 if(drag&&drag.new){const r=norm(drag);ctx.lineWidth=lw;ctx.strokeStyle=COLORS[kind];ctx.setLineDash([lw*3,lw*2]);
  ctx.strokeRect(r.x1,r.y1,r.x2-r.x1,r.y2-r.y1);ctx.setLineDash([]);}}
function norm(d){return{x1:Math.min(d.x0,d.x),y1:Math.min(d.y0,d.y),x2:Math.max(d.x0,d.x),y2:Math.max(d.y0,d.y)};}
function pt(e){const r=cv.getBoundingClientRect(),s=cv.width/r.width;return{x:Math.round((e.clientX-r.left)*s),y:Math.round((e.clientY-r.top)*s)};}
function hit(p){const bs=item().boxes;for(let i=bs.length-1;i>=0;i--){const b=bs[i];
 if(p.x>=b.x1&&p.x<=b.x2&&p.y>=b.y1&&p.y<=b.y2)return b;}return null;}
async function save(extra={}){const it=item();
 const res=await fetch(`/api/slice/${cur}`,{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({boxes:it.boxes,...extra})});const fresh=await res.json();
 Object.assign(it,fresh);if(sel)sel=it.boxes.find(b=>b.id===sel.id)||null;progress();}
cv.addEventListener('pointerdown',e=>{if(!drawMode&&e.pointerType!=='mouse')return;const p=pt(e);const b=hit(p);
 cv.setPointerCapture(e.pointerId);
 if(b){sel=b;setKind(b.kind,false);drag={move:b,x0:p.x,y0:p.y,orig:{...b}};}
 else{sel=null;drag={new:true,x0:p.x,y0:p.y,x:p.x,y:p.y};}draw();});
cv.addEventListener('pointermove',e=>{if(!drag)return;const p=pt(e);
 if(drag.new){drag.x=p.x;drag.y=p.y;}
 else{const dx=p.x-drag.x0,dy=p.y-drag.y0,o=drag.orig;Object.assign(drag.move,{x1:o.x1+dx,x2:o.x2+dx,y1:o.y1+dy,y2:o.y2+dy});drag.moved=true;}
 draw();});
cv.addEventListener('pointerup',async()=>{if(!drag)return;const d=drag;drag=null;
 if(d.new){const r=norm(d);if(r.x2-r.x1>=6&&r.y2-r.y1>=6){const b={id:'b'+Date.now().toString(36),kind,...r};
  item().boxes.push(b);await save();}}
 else if(d.moved)await save();draw();});
function setKind(k,apply=true){kind=k;document.querySelectorAll('[data-kind]').forEach(b=>b.classList.toggle('on',b.dataset.kind===k));
 if(apply&&sel){sel.kind=k;save().then(draw);}}
document.querySelectorAll('[data-kind]').forEach(b=>b.onclick=()=>setKind(b.dataset.kind));
async function delSel(){if(!sel)return;const it=item();it.boxes=it.boxes.filter(b=>b!==sel);sel=null;await save();draw();}
$('del').onclick=delSel;
$('scroll').onclick=()=>{drawMode=!drawMode;wrap.classList.toggle('draw',drawMode);
 $('scroll').textContent=drawMode?'Đang vẽ (chạm để cuộn)':'Đang cuộn (chạm để vẽ)';};
async function go(n){if(n<0||n>=state.slices.length)return;cur=n;showSlice();}
$('done').onclick=async()=>{await save({done:true});
 const next=state.slices.findIndex((s,i)=>i>cur&&!s.done);go(next>=0?next:Math.min(cur+1,state.slices.length-1));};
$('prev').onclick=()=>go(cur-1);
// Judge mode walks every labelled box on done slices that has no verdict yet.
let queue=[],qi=0;
function buildQueue(){queue=[];state.slices.forEach((s,i)=>{if(!s.done)return;
 for(const b of s.boxes)if(!s.verdicts[b.id])queue.push({i,b});});qi=0;}
const imgCache={};
function getImg(i,which){const k=i+which;
 if(!imgCache[k]){imgCache[k]=new Promise(r=>{const im=new Image();im.onload=()=>r(im);im.src=`/img/${i}/${which}`;});}return imgCache[k];}
async function showJudge(){if(qi>=queue.length){$('jinfo').textContent='Đã chấm hết các khung của những lát đã xong.';
  for(const c of [$('jo'),$('jc')]){c.width=1;c.height=1;}return;}
 const {i,b}=queue[qi];$('jinfo').textContent=`Khung ${qi+1}/${queue.length} · lát ${i+1} · ${NAMES[b.kind]}`;
 const [o,c]=await Promise.all([getImg(i,'original'),getImg(i,'clean')]);const pad=40;
 const x1=Math.max(0,b.x1-pad),y1=Math.max(0,b.y1-pad),x2=Math.min(o.naturalWidth,b.x2+pad),y2=Math.min(o.naturalHeight,b.y2+pad);
 for(const [cvs,im] of [[$('jo'),o],[$('jc'),c]]){cvs.width=x2-x1;cvs.height=y2-y1;
  const g=cvs.getContext('2d');g.drawImage(im,x1,y1,x2-x1,y2-y1,0,0,x2-x1,y2-y1);
  if(cvs.id==='jo'){g.strokeStyle=COLORS[b.kind];g.lineWidth=2;g.setLineDash([6,4]);g.strokeRect(b.x1-x1,b.y1-y1,b.x2-b.x1,b.y2-b.y1);}}}
async function judge(v){if(qi>=queue.length)return;const {i,b}=queue[qi];
 const res=await fetch(`/api/slice/${i}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({verdict:{box:b.id,verdict:v}})});
 Object.assign(state.slices[i],await res.json());qi++;showJudge();}
document.querySelectorAll('[data-verdict]').forEach(b=>b.onclick=()=>judge(b.dataset.verdict));
$('skip').onclick=()=>{qi++;showJudge();};
const pct=v=>v==null?'–':(v*100).toFixed(1)+'%';
async function showScore(){const s=await (await fetch('/api/score')).json();
 let h=`<p>Đã tính ${s.slices_done}/${s.slices_total} lát đã xong.${s.slices_changed?` ${s.slices_changed} lát bị đổi ảnh gốc nên bỏ qua.`:''}</p>
 <table><tr><th>Loại</th><th>Khung</th><th>App bắt được</th><th>Sạch đẹp</th><th>Còn chữ</th><th>Nền xấu</th><th>Chưa chấm</th></tr>`;
 for(const k of ['text','sfx','watermark']){const t=s.by_kind[k];
  h+=`<tr><td>${NAMES[k]}</td><td>${t.labels}</td><td>${t.found} (${pct(t.found_rate)})</td><td>${t.good} (${pct(t.good_rate)})</td><td>${t.text_left}</td><td>${t.bad_fill}</td><td>${t.unjudged}</td></tr>`;}
 h+=`</table><p>App vẽ ${s.app_boxes} khung, trong đó ${s.false_boxes} khung không trúng chữ nào bạn đánh dấu (bắt nhầm).</p>`;
 if(s.missed.length)h+=`<p>Bị bỏ sót: `+s.missed.map(m=>`<a href="#" data-go="${m.slice}">lát ${m.slice+1}</a>`).join(', ')+`</p>`;
 $('scorebox').innerHTML=h;
 $('scorebox').querySelectorAll('[data-go]').forEach(a=>a.onclick=e=>{e.preventDefault();setMode('mark');go(+a.dataset.go);});}
function setMode(m){mode=m;document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('on',b.dataset.mode===m));
 for(const id of ['mark','judge','score'])$(id).hidden=id!==m;
 if(m==='judge'){buildQueue();showJudge();}if(m==='score')showScore();if(m==='mark')draw();}
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>setMode(b.dataset.mode));
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;const k=e.key.toLowerCase();
 if(mode==='mark'){if(k==='1')setKind('text');else if(k==='2')setKind('sfx');else if(k==='3')setKind('watermark');
  else if(k==='delete'||k==='backspace'){e.preventDefault();delSel();}else if(k==='enter'){e.preventDefault();$('done').click();}
  else if(k==='a')go(cur-1);else if(k==='d')go(cur+1);}
 else if(mode==='judge'){if(k==='1')judge('good');else if(k==='2')judge('text_left');else if(k==='3')judge('bad_fill');else if(k==='s')$('skip').click();}});
load();
</script></body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
