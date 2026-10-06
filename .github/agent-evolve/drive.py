"""One-off: a long run that learns slides from zero, turns its skill into a plugin, evolves it, serves it over MCP, then uses it."""
import base64
import json
import os
from pathlib import Path
import shutil
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from app.agent.session import AgentSessionManager  # noqa: E402
from app.agent import sandbox  # noqa: E402
from app.agent.tools import Workspace  # noqa: E402
from app.ai_providers import resolve_provider  # noqa: E402

OUT = Path(sys.argv[1]).resolve()
BASE, MODEL, KEY = os.environ["AGENT_BASE"], os.environ["AGENT_MODEL"], os.environ["AGENT_API_KEY"]
OUT.mkdir(parents=True, exist_ok=True)
HOME = Path.home()
LIMITS = {"task1": 95, "task2": 125, "task3": 45, "final": 60}

TASK1 = ("/goal Học làm PowerPoint từ con số 0. Trong thư mục deck-lab/ của thư mục làm việc, tự viết code Python tạo một bộ slide 6–8 trang về một "
         "chủ đề bạn tự chọn. Đã cài sẵn python-pptx; LibreOffice và poppler có sẵn để xem kết quả: `soffice -env:UserInstallation=file:///tmp/lo_profile "
         "--headless --convert-to pdf --outdir OUT FILE.pptx` rồi `pdftoppm -png -r 50 FILE.pdf PREFIX` (dưới sandbox phải có -env:UserInstallation như vậy). "
         "Tự kiểm tra thật: mở lại file bằng python-pptx để đo chữ tràn khung, khung chồng nhau, cỡ chữ quá nhỏ, độ tương phản màu chữ và nền; "
         "nếu bạn xem được ảnh thì dùng view_image để nhìn các trang đã xuất. Sửa đến khi sạch lỗi. Sau đó viết skill pptx từ chính những gì đã học vào "
         ".agents/skills/pptx/SKILL.md (phần đầu có name và description; nội dung: bố cục và lưới, bảng màu, font và cỡ chữ, hiệu ứng chuyển trang, "
         "biểu đồ, lỗi hay gặp và cách tự kiểm tra), kèm script hỗ trợ trong .agents/skills/pptx/. Làm thêm ít nhất 2 bộ slide chủ đề khác chỉ theo skill, "
         "và nâng cấp skill sau mỗi bộ. Cuối cùng báo cáo đã học được gì.")
TASK2 = ("/goal Biến skill pptx thành plugin của chính bạn bằng tool plugin_write. Plugin cung cấp ít nhất hai tool: make_deck (nhận JSON gồm tiêu đề, "
         "từng slide với kiểu bố cục, nội dung, số liệu nếu có, và ghi ra một file .pptx áp đúng các quy tắc bạn đã học) và check_deck (kiểm tra một "
         "file .pptx: chữ tràn, chồng lấn, cỡ chữ, tương phản, căn lề, số chữ mỗi trang). Cắm vào rồi chỉ dùng chính các tool đó làm bộ slide mới, ít nhất "
         "3 chủ đề và 3 kiểu khác nhau (số liệu có biểu đồ, quy trình từng bước, so sánh hai cột). Sau mỗi vòng, tìm điểm yếu về bố cục, hiệu ứng "
         "(transition, animation), màu, chữ, cỡ chữ, rồi viết lại plugin bằng plugin_write để nó tiến hóa. Chỉ dừng khi check_deck không còn lỗi trên mọi "
         "bộ và bạn tự tin plugin đã thuần thục. Ghi nhật ký từng vòng tiến hóa vào deck-lab/EVOLUTION.md.")
TASK3 = ("/goal Biến plugin slide đó thành một MCP server. Có hai cách: tự viết một server stdio (JSON-RPC 2.0 với initialize, tools/list, tools/call) "
         "cho các tool make_deck và check_deck, hoặc dùng `python run.py mcp --folder . --write`, lệnh này phục vụ tool của mọi plugin qua MCP. "
         "Khai báo server trong .mcp.json của thư mục làm việc với tên deck. Tự kiểm tra thật: gửi initialize, tools/list và tools/call tới server, và "
         "tạo được một file .pptx qua MCP. Báo cáo cách cấu hình.")
FINAL = ("repo manga-translator có những thư mục file nào kiến trúc tổng quát sao và đi sâu vào từng lớp kiến trúc. "
         "Trình bày câu trả lời thành một bộ slide PowerPoint.")


def probe_vision() -> bool:
    """Whether the model reads an image: ask for the colour of a small red square."""
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (220, 20, 20)).save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    body = {"model": MODEL, "max_tokens": 20, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "What colour is this image? Answer with one word."}, {"type": "image_url", "image_url": {"url": url}}]}]}
    try:
        reply = requests.post(f"{BASE}/chat/completions", json=body, headers={"Authorization": f"Bearer {KEY}"}, timeout=120)
        text = reply.json()["choices"][0]["message"]["content"] if reply.ok else reply.text[:200]
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
    print("vision probe:", str(text)[:200], flush=True)
    return "red" in str(text).lower()


def run_phase(session, name: str, prompt: str) -> dict:
    """Send one prompt, approve what asks (as a person watching would), and wait up to the phase's limit."""
    start = time.time()
    session.set_deadline(LIMITS[name] * 60)
    first = len(session.events)
    session.send(prompt)
    seen, approvals, last_print = first, [], start
    while True:
        time.sleep(3)
        for event in session.events[seen:]:
            seen += 1
            if event["type"] == "tool":
                print(f"[{name} {int(time.time() - start)}s] {event['name']} {'ok' if event.get('ok') else 'FAILED'}: {str(event.get('output', ''))[:140]!r}", flush=True)
            elif event["type"] in ("notice", "error"):
                print(f"[{name}] {event['type']}: {event.get('text', '')[:300]}", flush=True)
        if session.status == "waiting" and session.pending:
            approvals.append(session.pending["name"])
            print(f"[{name}] approving {session.pending['name']}", flush=True)
            session.decide("allow")
        if session.status == "idle" and time.time() - start > 10:
            break
        if time.time() - start > (LIMITS[name] + 10) * 60:
            session.stop()
            time.sleep(20)
            break
        if time.time() - last_print > 600:
            last_print = time.time()
            print(f"[{name}] still working, {int((time.time() - start) / 60)} min, usage {session.usage}", flush=True)
    events = session.events[first:]
    with open(OUT / f"{name}-events.jsonl", "w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    tools = [e["name"] for e in events if e["type"] == "tool"]
    return {"minutes": round((time.time() - start) / 60, 1), "status": session.status, "approvals": approvals,
            "tool_counts": {t: tools.count(t) for t in sorted(set(tools))},
            "failed_tools": sum(1 for e in events if e["type"] == "tool" and e.get("ok") is False),
            "errors": [e.get("text", "")[:300] for e in events if e["type"] == "error"],
            "final": next((e["text"] for e in reversed(events) if e["type"] == "assistant" and e.get("text")), "")[:8000]}


def main() -> None:
    vision = probe_vision()
    folder = HOME / ".manga-agent"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "profile.json").write_text(json.dumps({"token_budget": 10 ** 12, "max_steps": 5000, "goal_turns": 300, "vision": vision}))
    label = os.environ.get("AGENT_LABEL", "Provider")
    provider = resolve_provider(label.lower(), label=label, protocol="openai", api_base=BASE)
    manager = AgentSessionManager(OUT.parent / "store", home=HOME)
    policy = sandbox.Policy("workspace-write", True)
    report = {"model": MODEL, "vision": vision, "sandbox": sandbox.backend()}

    session = manager.create(provider, KEY, MODEL, Workspace(ROOT, policy), "auto")
    for name, prompt in (("task1", TASK1), ("task2", TASK2), ("task3", TASK3)):
        report[name] = run_phase(session, name, prompt)
        report[name]["usage"] = dict(session.usage)
        (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    report["plugins_after_task3"] = session.command("/plugins")["message"]
    agent_rows = [r.id for r in session.kernel.rows.values() if r.source == "agent"]
    session.close()

    # A fresh session where the slide tools exist only through MCP: the agent's plugin rows are switched off, its .mcp.json servers trusted.
    final = manager.create(provider, KEY, MODEL, Workspace(ROOT, policy), "auto")
    for row in agent_rows:
        final.command(f"/plugins disable {row}")
    final._ensure_mcp()
    for name, status in list(final.mcp_status.items()):
        if status["scope"] == "workspace":
            final.trust_mcp(name)
    report["final_setup"] = {"plugins": final.command("/plugins")["message"], "mcp": final.command("/mcp")["message"],
                             "mcp_tools": sorted(final.mcp_tools)}
    print("final setup:", json.dumps(report["final_setup"], ensure_ascii=False)[:2000], flush=True)
    report["final"] = run_phase(final, "final", FINAL)
    report["final"]["usage"] = dict(final.usage)
    report["final"]["mcp_calls"] = sum(1 for e in final.events if e["type"] == "tool" and e["name"].startswith("mcp__"))
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    manager.close_all()

    # Keep what the agent made: decks, its skill, its plugins, its MCP config, and pictures of every deck.
    keep = OUT / "made"
    for source in ("deck-lab", ".agents/skills/pptx", ".mcp.json"):
        path = ROOT / source
        if path.is_dir():
            shutil.copytree(path, keep / source, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pdf"))
        elif path.is_file():
            (keep / source).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, keep / source)
    if (folder / "plugins").is_dir():
        shutil.copytree(folder / "plugins", keep / "plugins", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    print(json.dumps({k: (v if not isinstance(v, dict) else {x: y for x, y in v.items() if x != "final"}) for k, v in report.items()},
                     ensure_ascii=False, indent=1)[:6000], flush=True)


if __name__ == "__main__":
    main()
