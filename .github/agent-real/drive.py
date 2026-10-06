"""One-off: type `manga` inside the codex and omp clones like a user, save the key in Settings, and ask the Agent tab about each folder."""
import json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path

APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP / "scripts"))
import install  # noqa: E402

OUT = Path(os.environ["OUT"]).resolve()
FOLDERS = {"codex": Path(os.environ["CODEX"]).resolve(), "omp": Path(os.environ["OMP"]).resolve()}
MODEL, KEY = os.environ["AGENT_MODEL"], os.environ["AGENT_API_KEY"]
BASE = "https://api.minirouter.sh/v1"
MINUTES = float(os.environ.get("MINUTES_EACH", "20"))
HEAD = {"X-Manga-Agent": "1"}
OUT.mkdir(parents=True, exist_ok=True)
PROMPT = ("Thư mục này là dự án gì? Hãy tự xem cấu trúc thư mục và đọc code rồi trả lời ngắn gọn: "
          "(1) dự án làm gì, viết bằng ngôn ngữ gì; (2) vòng lặp agent chính (gửi model → nhận lời gọi tool → chạy tool → gửi kết quả lại) nằm ở file nào, hàm nào; "
          "(3) lệnh shell của agent được chạy an toàn thế nào (sandbox, xin phép). Dẫn file:dòng cụ thể cho từng ý. Không sửa file nào.")

# The `manga` command exactly as the installer writes it, and a browser that only records the page it was asked to open.
bin_dir = OUT.parent / "bin"
bin_dir.mkdir(exist_ok=True)
(bin_dir / "manga").write_text(install.launcher_text(Path(sys.executable), APP / "run.py"), encoding="utf-8")
(bin_dir / "manga").chmod(0o755)
urls = OUT.parent / "urls.txt"
(bin_dir / "browser").write_text(f'#!/bin/sh\nprintf "%s\\n" "$1" >> "{urls}"\n', encoding="utf-8")
(bin_dir / "browser").chmod(0o755)
env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "BROWSER": f"{bin_dir / 'browser'} %s"}
env.pop("AGENT_API_KEY", None)
log = open(OUT / "server.log", "w", encoding="utf-8")


def type_manga(folder: Path) -> subprocess.Popen:
    """Run `manga` in a terminal (a pty, so it counts as typed) whose working folder is ``folder``."""
    return subprocess.Popen(["script", "-qfec", "manga", "/dev/null"], cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)


def next_url(count: int, timeout: float = 240) -> str:
    end = time.time() + timeout
    while time.time() < end:
        lines = urls.read_text().split() if urls.exists() else []
        if len(lines) >= count:
            return lines[count - 1]
        time.sleep(1)
    raise SystemExit(f"manga did not open a page ({count})")


def api(url: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEAD), timeout=30) as response:
        return json.load(response)


server = type_manga(FOLDERS["codex"])
report = {"launch": {}}
from playwright.sync_api import sync_playwright  # noqa: E402

with sync_playwright() as p:
    local = os.environ.get("CHROMIUM_PATH")
    browser = p.chromium.launch(executable_path=local, headless=False, args=["--headless=new"]) if local else p.chromium.launch()
    context = browser.new_context(viewport={"width": 1366, "height": 1000})
    shots = 0

    def shot(page, name: str) -> None:
        global shots
        shots += 1
        page.wait_for_timeout(800)
        page.screenshot(path=str(OUT / f"{shots:02d}-{name}.png"))

    for index, (name, folder) in enumerate(FOLDERS.items(), 1):
        if index > 1:
            type_manga(folder).wait(timeout=120)
        url = next_url(index)
        origin = url.split("/?")[0]
        report["launch"][name] = {"url_has_folder": "workspace=" in url}
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda e, errors=errors: errors.append(str(e)))
        page.goto(url)
        page.wait_for_load_state("networkidle")
        if index == 1:
            page.click("#settings-toggle")
            page.locator(".ai-custom-provider-form > summary").click()
            page.get_by_label("Tên provider tùy chỉnh").fill("minirouter")
            page.get_by_label("Mã provider tùy chỉnh").fill("minirouter")
            page.get_by_label("API root HTTPS của provider").fill(BASE)
            page.get_by_label("Model vision của provider tùy chỉnh").fill(MODEL)
            page.get_by_label("API key của provider tùy chỉnh").fill(KEY)
            page.get_by_role("button", name="Lưu provider").click()
            page.wait_for_timeout(3000)
            shot(page, "settings-saved")
            page.click("#settings-close")
        page.click('.sidebar-link[data-route="agent"]')
        page.wait_for_function("document.getElementById('agent-workspace').value !== ''", timeout=30000)
        seen_folder = page.eval_on_selector("#agent-workspace", "e => e.value")
        report["launch"][name].update(workspace=seen_folder, matches=Path(seen_folder) == folder,
                                      placeholder=page.eval_on_selector("#agent-input", "e => e.placeholder"))
        try:
            page.wait_for_selector('#agent-provider option[value="minirouter"]', state="attached", timeout=30000)
        except Exception:
            shot(page, f"{name}-no-provider")
            report["launch"][name]["status"] = page.inner_text("#agent-status")
            continue
        page.select_option("#agent-provider", "minirouter")
        page.fill("#agent-model", MODEL)
        page.select_option("#agent-mode", "auto")
        page.select_option("#agent-sandbox", "read-only")
        shot(page, f"{name}-ready")
        before = {s["id"] for s in api(f"{origin}/api/agent/sessions")["sessions"]}
        page.fill("#agent-input", PROMPT)
        page.press("#agent-input", "Enter")
        start, sid, snap, shot_tools = time.time(), None, {}, 0
        while time.time() - start < MINUTES * 60:
            time.sleep(3)
            if sid is None:
                fresh = [s["id"] for s in api(f"{origin}/api/agent/sessions")["sessions"] if s["id"] not in before]
                sid = fresh[0] if fresh else None
                continue
            snap = api(f"{origin}/api/agent/sessions/{sid}")
            tools = [e for e in snap["events"] if e["type"] == "tool"]
            if len(tools) >= (shot_tools + 1) * 4 and shot_tools < 3:
                shot_tools += 1
                page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
                shot(page, f"{name}-tools{shot_tools}")
            if snap["status"] == "idle" and time.time() - start > 20:
                break
        if sid and snap.get("status") != "idle":
            urllib.request.urlopen(urllib.request.Request(f"{origin}/api/agent/sessions/{sid}/stop", method="POST", headers=HEAD), timeout=30)
            time.sleep(10)
            snap = api(f"{origin}/api/agent/sessions/{sid}")
        page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
        shot(page, f"{name}-end")
        events = snap.get("events", [])
        with open(OUT / f"{name}-events.jsonl", "w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        tools = [e["name"] for e in events if e["type"] == "tool"]
        report[name] = {"status": snap.get("status"), "minutes": round((time.time() - start) / 60, 1), "usage": snap.get("usage"),
                        "tool_counts": {t: tools.count(t) for t in sorted(set(tools))},
                        "failed_tools": [(e["name"], e.get("output", "")[:200]) for e in events if e["type"] == "tool" and e.get("ok") is False],
                        "errors": [e.get("text", "")[:300] for e in events if e["type"] == "error"],
                        "notices": [e.get("text", "")[:200] for e in events if e["type"] == "notice"][:40],
                        "page_errors": errors[:10],
                        "final": next((e["text"] for e in reversed(events) if e["type"] == "assistant" and e.get("text")), "")[:6000]}
    browser.close()

(OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: (v if k == "launch" else {x: y for x, y in v.items() if x not in ("final", "notices")}) for k, v in report.items()},
                 ensure_ascii=False, indent=1))
os.killpg(server.pid, signal.SIGTERM)
