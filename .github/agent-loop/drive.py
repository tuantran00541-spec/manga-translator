"""One-off: a long real-model run through the Agent screen, with screenshots when it searches, reads, uses a skill, edits or tests."""
import json, os, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import uvicorn
from app.agent.session import AgentSessionManager
from app.ai_providers import resolve_provider
from app.routers import agent as router
from app.main import app

OUT, WORK = Path(sys.argv[1]), Path(sys.argv[2])
MINUTES = float(sys.argv[3])
BASE, MODEL, LABEL = os.environ["AGENT_BASE"], os.environ["AGENT_MODEL"], os.environ.get("AGENT_LABEL", "Provider")
OUT.mkdir(parents=True, exist_ok=True)
(WORK / "project").mkdir(parents=True, exist_ok=True)
PROMPT = ("/goal Dùng web_search tìm một dự án Python mã nguồn mở đang hot trên GitHub dạo gần đây (trending), có test bằng pytest và đủ nhỏ để chạy trong vài phút. "
          "Đọc trang dự án bằng web_fetch rồi clone nó vào thư mục làm việc, đọc README, cài trong một venv và chạy test. "
          "Sau đó chọn một issue mở nhỏ hoặc một chỗ cải thiện nhỏ trong 1–3 file: đọc issue/tài liệu trên web nếu cần, dùng skill phù hợp (ví dụ test-driven-development, systematic-debugging), "
          "sửa code, thêm hoặc sửa test, chạy test đến khi xanh. Cuối cùng báo cáo: dự án nào, vì sao chọn, đã sửa gì (kèm git diff --stat), kết quả test thật.")

home = Path.home()
(home / ".manga-agent").mkdir(parents=True, exist_ok=True)
(home / ".manga-agent" / "profile.json").write_text(json.dumps({"token_budget": 10 ** 12, "max_steps": 3000}))
mgr = AgentSessionManager(home / ".manga-agent", home=home)
router.agent_sessions = mgr
provider = resolve_provider("polargrid", label=LABEL, protocol="openai", api_base=BASE)
key = os.environ["AGENT_API_KEY"]


async def real_provider(provider_id):
    return provider, key

router._provider_and_key = real_provider
server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8766, log_level="warning"))
threading.Thread(target=server.run, daemon=True).start()
for _ in range(120):
    try:
        import urllib.request
        urllib.request.urlopen("http://127.0.0.1:8766/", timeout=1)
        break
    except Exception:
        time.sleep(0.5)

LABELS = (("web_search", "web-search"), ("web_fetch", "web-fetch"), ("skill", "skill"), ("todo_write", "plan"), ("spawn_agent", "helper"),
          ("fan_out", "helpers"), ("run_script", "script"), ("edit_file", "edit"), ("apply_patch", "edit"), ("edit_lines", "edit"), ("write_file", "write"))
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    local = os.environ.get("CHROMIUM_PATH")
    browser = p.chromium.launch(executable_path=local, headless=False, args=["--headless=new"]) if local else p.chromium.launch()
    page = browser.new_page(viewport={"width": 1366, "height": 1000})
    page.route("**/api/visual_qc/settings", lambda r: r.fulfill(json={"providers": {"polargrid": {"id": "polargrid", "label": LABEL, "configured": True}}}))
    page.route("**/api/visual_qc/providers/*/models", lambda r: r.fulfill(json={"models": [MODEL]}))
    page.goto("http://127.0.0.1:8766/")
    page.click('.sidebar-link[data-route="agent"]')
    page.wait_for_selector("#agent-provider option", state="attached", timeout=30000)
    page.fill("#agent-model", MODEL)
    page.select_option("#agent-mode", "auto")
    page.select_option("#agent-sandbox", "workspace-write")
    page.check("#agent-network")
    page.evaluate("w => { document.querySelector('#agent-workspace').value = w; }", str(WORK / "project"))
    page.fill("#agent-input", PROMPT)
    page.press("#agent-input", "Enter")
    shots, seen, calls = 0, set(), {}
    start = last_periodic = time.time()

    def shot(name: str, call_id: str = "") -> None:
        global shots
        if shots >= 45:
            return
        shots += 1
        try:
            page.wait_for_timeout(1200)
            row = page.locator(f'.agent-row[data-call-id="{call_id}"]') if call_id else None
            if row is not None and row.count():
                row.first.evaluate("e => { const g = e.closest('.agent-group'); if (g) g.open = true; e.open = true; e.scrollIntoView({block: 'center'}); }")
            else:
                page.evaluate("""() => { document.querySelectorAll('*').forEach(e => {
                    const y = getComputedStyle(e).overflowY;
                    if ((y === 'auto' || y === 'scroll') && e.scrollHeight > e.clientHeight + 10) e.scrollTop = e.scrollHeight; });
                    window.scrollTo(0, document.body.scrollHeight); }""")
            page.wait_for_timeout(500)
            page.screenshot(path=str(OUT / f"{shots:02d}-{name}.png"))
        except Exception as exc:
            print("screenshot failed:", exc)

    session = None
    while time.time() - start < MINUTES * 60:
        time.sleep(2)
        if session is None:
            session = next(iter(mgr.sessions.values()), None)
            continue
        for event in list(session.events):
            if event["type"] == "assistant":
                for c in event.get("calls") or []:
                    calls[c["id"]] = c
            elif event["type"] == "tool" and event["id"] not in seen:
                seen.add(event["id"])
                c = calls.get(event["id"], {"name": event["name"], "args": {}})
                command = str((c.get("args") or {}).get("command", ""))
                label = next((l for n, l in LABELS if n == event["name"]), "")
                if event["name"] == "run_command":
                    label = "clone" if "git clone" in command else "tests" if "pytest" in command else "install" if "pip install" in command else ""
                if label and sum(1 for f in OUT.glob(f"*-{label}*.png")) < 3:
                    shot(f"{label}-{event['name']}", event["id"])
            elif event["type"] == "notice" and "tóm gọn" in event.get("text", "") and event.get("seq") not in seen:
                seen.add(event.get("seq"))
                shot("compaction")
        if time.time() - last_periodic > 20 * 60:
            last_periodic = time.time()
            shot(f"t{int((time.time() - start) / 60)}m")
        if session.status == "idle" and time.time() - start > 60:
            break
    if session is not None and session.status != "idle":
        session.stop()
        for _ in range(90):
            if session.status == "idle":
                break
            time.sleep(2)
    shot("end")
    if session is not None:
        with open(OUT / "events.jsonl", "w", encoding="utf-8") as handle:
            for event in session.events:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        tools = [e["name"] for e in session.events if e["type"] == "tool"]
        summary = {"status": session.status, "minutes": round((time.time() - start) / 60, 1), "usage": session.usage, "stats": session.stats,
                   "tool_counts": {n: tools.count(n) for n in sorted(set(tools))},
                   "errors": [e["text"][:300] for e in session.events if e["type"] == "error"],
                   "notices": [e["text"][:200] for e in session.events if e["type"] == "notice"][:80],
                   "final": next((e["text"] for e in reversed(session.events) if e["type"] == "assistant" and e.get("text")), "")[:6000]}
        (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in summary.items() if k not in ("notices", "final")}, ensure_ascii=False))
    browser.close()
