"""One-off: grade hand-labelled translations with Jev to see where good and bad lines fall on each score."""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from app.ai_mode.polish import LIMITS, QUESTIONS, Line, line_state

# Source, then good, stiff (word for word), wrong meaning and garbled Vietnamese.
CASES = [
    ("Get out of here, now!", "Cút khỏi đây ngay!", "Hãy đi ra khỏi nơi này ở ngay bây giờ!", "Vào đây ngay!",
     "Ngay ra đây khỏi, cút đi nó!"),
    ("I won't let you die.", "Tôi sẽ không để cậu chết đâu.", "Tôi sẽ không cho phép bạn để chết.", "Tôi sẽ để cậu chết.",
     "Chết không để tôi cậu sẽ."),
    ("What the hell is going on here?!", "Chuyện quái gì đang xảy ra vậy?!", "Cái địa ngục gì đang đi tiếp ở đây?!",
     "Ở đây yên bình quá nhỉ?!", "Đang quái ra sao gì đây vậy chuyện?!"),
    ("You're stronger than you look.", "Cậu mạnh hơn vẻ ngoài đấy.", "Bạn là mạnh hơn so với bạn nhìn.",
     "Cậu yếu hơn tôi nghĩ.", "Mạnh vẻ cậu ngoài hơn đấy trông."),
    ("Don't worry. I'll handle it.", "Đừng lo. Để tôi lo.", "Không lo lắng. Tôi sẽ xử lý nó.", "Lo đi. Tôi mặc kệ đấy.",
     "Lo đừng. Tôi để nó xử."),
    ("Thanks to you, we survived.", "Nhờ có cậu mà bọn tôi mới sống sót.", "Cảm ơn đến bạn, chúng tôi đã sống sót.",
     "Vì cậu mà bọn tôi suýt chết.", "Sống nhờ sót cậu tôi bọn mà."),
    ("It's been a long time, old friend.", "Lâu rồi không gặp, bạn cũ.", "Nó đã là một thời gian dài, người bạn già.",
     "Mới gặp hôm qua mà, người lạ.", "Gặp lâu bạn rồi không cũ."),
    ("The gate will open at midnight.", "Cổng sẽ mở lúc nửa đêm.", "Cái cổng sẽ được mở ra tại thời điểm nửa đêm.",
     "Cổng sẽ đóng lúc bình minh.", "Nửa mở cổng sẽ đêm lúc."),
    ("Are you out of your mind?!", "Cậu điên rồi à?!", "Bạn có ở bên ngoài tâm trí của bạn không?!",
     "Cậu thông minh thật đấy?!", "Rồi điên à cậu à?!"),
    ("I'll never forgive you for this.", "Tôi sẽ không bao giờ tha thứ cho ngươi chuyện này.",
     "Tôi sẽ không bao giờ tha thứ bạn cho cái này.", "Tôi tha thứ cho ngươi rồi.",
     "Tha bao giờ không ngươi tôi này chuyện."),
]
KINDS = ("good", "stiff", "wrong", "garbled")


def grade(item):
    source, kind, text = item
    state = line_state(Line(0, "x", source, text))
    response = requests.post(os.environ["JUDGE_URL"], headers={"Authorization": "Bearer " + os.environ["KEY"]},
                             json={"model": "typesafe-ai/jev", "state": state, "questions": QUESTIONS}, timeout=(10, 60))
    body = response.json() if response.ok else {"error": response.text[:300]}
    scores = {k: v.get("score") for k, v in (body.get("answers") or {}).items() if isinstance(v, dict)}
    return {"source": source, "kind": kind, "text": text, "scores": scores, "status": response.status_code,
            **({"error": body["error"]} if "error" in body else {})}


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    items = [(case[0], kind, text) for case in CASES for kind, text in zip(KINDS, case[1:])]
    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(grade, items))
    summary = {}
    for kind in KINDS:
        got = [r["scores"] for r in rows if r["kind"] == kind and r["scores"]]
        summary[kind] = {q: sorted(round(s[q], 2) for s in got if s.get(q) is not None) for q in QUESTIONS}
        summary[kind]["flagged"] = sum(any(s.get(q, 9) < limit for q, limit in LIMITS.items()) for s in got)
    (out / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0 if all(r["status"] == 200 for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
