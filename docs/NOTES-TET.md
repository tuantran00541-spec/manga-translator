# Ghi chú bàn giao (21/10/2026)

## Trạng thái
- Main có đủ app, A.I mode, hướng dẫn cài tiện ích và tab Chaos (PR #85, merge 5393d57f).
- Bản phát hành mới nhất vẫn là V0.3.1. V0.4.0 chưa phát hành.
- Models lấy từ release `models-v1` (xem `scripts/install.py`).
- SonarCloud đỏ trên main từ trước. Phần lớn cảnh báo ở `app/agent/` là chủ ý: agent chạy lệnh và đọc đường dẫn do model chọn, router chỉ nhận yêu cầu từ máy này.

## Terminal-Bench 2.1 (harness hiện tại, 89 bài)
"Sạch" không tính bài mà agent chạm tới nguồn của benchmark.

| Model | Đạt | Sạch | Harness cũ |
|---|---|---|---|
| DeepSeek V4.1 Flash (River) | 72 | 71 | 71 |
| Beam (CommandCode) | 64 | 61 | 42/59 |
| DeepSeek, bản cắt/bàn giao/1M | 71 | 71 | 71 |
| DeepSeek, prompt tiếng Anh mới | 72 | 71 | |

A/B trên Beam (2 lượt mỗi nhánh): đủ công cụ 64/61 và 64/63; chỉ shell 60/60 và 67/67; shell kèm danh sách lệnh 64/61 và 65/61. Chênh lệch nằm trong nhiễu nên tùy chọn `shell_only` đã bị xóa.
Thêm 3 skill bên ngoài (karpathy-guidelines, systematic-debugging, verification-before-completion): 63/59 và 63/61, nhưng agent không đọc SKILL.md nào trong cả 2 lượt, nên danh sách skill trong prompt không đổi gì.

Kết quả nằm ở `audit-results/terminal-bench-2-1/` trên nhánh `audit-evidence`. Mỗi trial có `events.jsonl.gz`, `result.json` và `verifier/`.

## Chi phí River
| Lượt | USD |
|---|---|
| 39404992762 | 5.35 |
| 39430009322 (shard 1 chạy lại) | 5.42 |
| 39441430048 | 6.00 |
| 39521835269 (prompt mới) | 6.30 |

Tổng River: 23,07 USD; số dư còn khoảng 1,9 USD. `max_usd` giờ là giới hạn cho cả lượt, chia đều cho các shard.

## Lỗi đã biết
- Quota tuần của CommandCode hết lúc 21/10 01:21Z, reset 24/10 23:23Z; lượt Muse Spark 1.3 bị hủy vì thế.
- Bài `sanitize-git-repo` có token giả nên push protection chặn lưu. `save-evidence` đã che token từ commit 6523f472; các lượt cũ hơn mất shard 1.
- Runner GitHub đôi khi không bật Docker, khiến shard đó ra 0 bài. Chạy lại job đó là được.
- `/etc/hosts` không chặn github.com, nên agent vẫn có thể mở repo của benchmark trên GitHub.
- Popup của tiện ích bật lại sau 120 giây.

## Ý tưởng chưa làm
- Adapter `dsh` cho harness của DeepSeek.
- Relay giảm tốc dần khi gặp 429 (AIMD), chưa được duyệt.
