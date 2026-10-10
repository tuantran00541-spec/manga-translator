# Agent Mode — Safe Operation Notes

Ghi chú vận hành an toàn cho Agent mode (`app/agent/`, API tại `app/routers/agent.py`).
Safe-operation notes for Agent mode.

## 1. Agent API là local-only, không có auth

- Router Agent chỉ chấp nhận kết nối **loopback** (`app/routers/agent.py`: `guard`), và yêu cầu header `X-Manga-Agent: 1`.
- **Không có auth theo user/session**: bất kỳ ai gọi được tới port của app đều có thể điều khiển agent (đọc/ghi file trong workspace, chạy lệnh sandbox, dùng API key của bạn cho model).
- Vì vậy **không expose app ra LAN hay reverse proxy** (ví dụ bind `0.0.0.0`, `MANGA_ALLOWED_HOSTS`, nginx forward). Agent mode được thiết kế để chạy trên máy của bạn và chỉ bạn dùng.
- Kill switch: đặt `MANGA_AGENT_MODE=0` để tắt hoàn toàn Agent API (trả 404).

## 2. MCP server được trust sẽ thấy toàn bộ `os.environ`

- MCP stdio server ở scope workspace chỉ khởi động sau khi bạn **trust** (trust được pin theo digest của config, `app/agent/mcp.py`, `app/agent/context.py`).
- Nhưng một khi đã trust, server đó **kế thừa toàn bộ `os.environ` của tiến trình app** — gồm cả API key của các AI provider (`mcp.py`: `env = {**os.environ, **config env}`). Khác với lệnh chạy trong sandbox (bị `clean_env()` lọc secret), MCP server không bị lọc env.
- Chỉ trust MCP server mà bạn tin tưởng (do bạn tự viết hoặc từ nguồn đáng tin). Trust được pin theo digest của config: nếu bạn sửa config của server (command/args/env), digest đổi và trust cũ mất hiệu lực, phải trust lại. Muốn thu hồi trust, xóa entry tương ứng trong `data/agent/trust.json` rồi restart session.
