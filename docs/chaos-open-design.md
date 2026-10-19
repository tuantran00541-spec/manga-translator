# Chaos mở — bản thiết kế

Mục tiêu: lõi giữ nhỏ như hiện tại, mọi thứ thêm vào đi qua ba cửa mở:

1. **MCP**: người dùng gắn server nào cũng được.
2. **Plugin**: người dùng hoặc chính model viết.
3. **Provider**: chạy với model nào cũng được.

## 1. Lõi (không đổi)

**Lõi gồm:**
- Vòng lặp phiên.
- Tool có sẵn: file, sửa, lệnh, web, subagent.
- 3 chế độ duyệt.
- Sandbox.
- Lưu phiên.
- Tóm tắt ngữ cảnh.
- Skill.

**Lõi không biết gì về plugin cụ thể nào.** Plugin không được:
- đổi chế độ duyệt;
- đọc key;
- tắt sandbox;
- sửa một lệnh đã được duyệt.

Đây là các test bảo mật hiện có và giữ nguyên.

## 2. MCP tùy ý

**Đã có:**
- stdio và http.
- Dán JSON kiểu Claude Code hoặc Cursor.
- Bật/tắt và thử kết nối.
- `.mcp.json` của dự án phải được tin cậy theo digest.
- Mục `replace` để thay tool có sẵn bằng tool MCP.

**Thêm:**

| Mục | Cách làm |
|---|---|
| Lọc tool theo server | `"tools": {"allow": [...], "deny": [...]}` trong cấu hình. Tool bị lọc không gửi cho model, nên tiết kiệm ngữ cảnh. |
| Kiểu tool → chế độ duyệt | `"kinds": {"search_docs": "read", "deploy": "exec"}`. Tool không khai báo kiểu thì coi là `exec` (luôn qua duyệt như lệnh). |
| Prompts của server | Mỗi prompt thành một lệnh `/mcp:server:prompt`. |
| Resources của server | Tool `mcp_resource(server, uri)` đọc resource. Resource được coi là nội dung ngoài, nên vẫn qua taint guard. |
| Ảnh trả về | Trả cho model nếu model nhìn được ảnh; nếu không thì ghi "[image omitted]" như hiện nay. |
| `notifications/tools/list_changed` | Tải lại danh sách tool của server đó, không phải khởi động lại. |
| Timeout và khởi động lại | `"timeout"` cho từng server. Server chết giữa phiên thì tự khởi động lại một lần, sau đó báo lỗi. |
| UI | Bảng MCP hiện danh sách tool của từng server, mỗi tool có công tắc riêng. Bấm công tắc thì ghi vào `allow`/`deny`. |

## 3. Plugin — hai tầng

Quyết định chính: model tự viết tool thì tool đó **mặc định là MCP server chạy trong sandbox**, không chạy trong tiến trình.

### Tầng 1: tool server (model viết thoải mái)

**Model viết:**
- Model dùng tool `extend` (một tool có sẵn mới) để tạo `.agents/extensions/<tên>/`.
- Trong đó có `server.py` dùng SDK nhỏ `chaos_ext`, ví dụ:
  ```python
  @tool(kind="read")
  def f(x: str) -> str: ...
  ```
- Kèm `manifest.json` gồm tên, mô tả, và kiểu các tool.

**Harness kiểm tra rồi mới bật:**
- Chạy server **trong sandbox của workspace** qua stdio, đúng như một MCP server.
- Gọi `tools/list` để kiểm tra.
- Nếu lỗi, trả lỗi về cho model sửa.
- Nếu ổn, tool hiện ngay trong lượt sau với tên `ext__<tên>__<tool>`.

**Duyệt:**
- **Tạo hoặc sửa extension:** hỏi bạn một lần ở chế độ Hỏi mọi thứ và Tự sửa. Card duyệt hiện diff.
- **Gọi tool:** duyệt theo `kind` như mọi tool khác.
- **Lý do an toàn:** code chạy trong sandbox giống `run_command`, nên rủi ro không lớn hơn để model tự chạy script.
- **Lưu và chia sẻ:**
  - Commit chung với dự án được.
  - Người khác mở dự án thì phải tin cậy theo digest, giống `.mcp.json`.

### Tầng 2: plugin trong tiến trình (`register(api)`)

**Dùng khi nào:** cần móc vào harness, chứ không chỉ thêm tool:
- hook;
- lệnh `/`;
- thêm prompt;
- provider model mới;
- vòng lặp riêng.

**Nguồn và tin cậy:**
- Nằm ở `~/.manga-agent/plugins/` hoặc `.agents/plugins/`, như hiện nay.
- Chỉ chạy khi bạn tin cậy đúng digest.

**Đổi:** plugin không nhận `session` thô nữa mà nhận một đối tượng `api` cố định, có `api.version = 1`.

```python
def register(api):
    api.tool(spec, handler, kind="read")          # handler(ctx, args) -> str
    api.command("name", "mô tả", handler)          # /name
    api.on("pre_tool", fn)    # chặn: trả chuỗi lý do
    api.on("post_tool", fn)   # thêm chữ vào kết quả
    api.on("session_start" | "turn_end" | "before_model" | "after_model", fn)
    api.prompt(fn)                                 # thêm đoạn vào system prompt
    api.provider("protocol_name", adapter)         # xem mục 4
    api.loop("name", fn)
    api.config     # mục "plugins": {"<tên>": {...}} trong profile.json
    api.state      # kho JSON riêng của plugin
    api.log(msg)
```

**`ctx` mà handler nhận**, chỉ gồm những thứ đọc được:
- workspace root;
- `ctx.run(cmd)`: chạy lệnh qua sandbox và qua duyệt;
- `ctx.read(path)`;
- `ctx.ask_user(q)`;
- `ctx.emit(kind, data)`.

`ctx` không có key, không có `mode` và không có `registry`.

**Model viết plugin tầng 2:**
- **Được viết:** qua `extend` với `"inprocess": true`.
- **Luôn hỏi bạn,** ở mọi chế độ, kể cả Không duyệt.
- **Card duyệt cảnh báo rõ** rằng code này chạy với quyền của harness.
- **Duyệt xong:** digest được ghi lại và plugin nạp nóng (`/plugins reload`).

**Lỗi nạp:** harness trả về traceback cho model.

## 4. Mọi model

**Đã có:**
- Giao thức kiểu OpenAI. Cái này phủ Ollama, LM Studio, vLLM, llama.cpp, OpenRouter, DeepSeek, Groq...
- Giao thức Gemini.
- Tự lùi về text tool call khi model không hỗ trợ tool native.
- Nhiều key, fallback model.

**Thêm:**

| Mục | Cách làm |
|---|---|
| Giao thức Anthropic Messages | Adapter thứ ba trong `client.py`, gồm tool use, ảnh, thinking và prompt cache. |
| Adapter từ plugin | `api.provider(name, adapter)`, với `adapter(request) -> {"text", "calls", "usage"}` theo đúng định dạng nội bộ mà `client.complete` trả. Provider tùy chỉnh có thể chọn `protocol` là tên này. |
| Bảng năng lực model | `profile.json` → `models.<id>` khai báo: `tools` (native/text), `vision`, `reasoning`, `context_window`, `max_output`. Mục nào thiếu thì đoán từ `models.py`, và đoán sai thì tự sửa theo lỗi lần đầu (đã làm cho context window). |
| Model yếu | Đặt `"tools": "lean"` thì chỉ gửi tool cốt lõi (read, edit, run, search). Tool MCP và extension chỉ được gửi khi model gọi `tool_search` (tìm theo tên hoặc mô tả). |
| Prompt trung lập | System prompt không giả định hãng. Hướng dẫn riêng cho từng model đặt ở `models.<id>.prompt` hoặc trong plugin. |

## 5. Thứ tự làm

Mỗi bước có test, 4 script sanity và ảnh chụp UI, rồi mới sang bước sau.

1. **API plugin v1:**
   - Đối tượng `api`/`ctx` thay cho `session` thô.
   - Thêm 4 sự kiện mới.
   - `api.config` và `api.state`.
   - Test bảo mật: plugin không với tới key, mode, registry.
2. **Extension tầng 1:**
   - Tool `extend` và SDK `chaos_ext`.
   - Chạy server trong sandbox, kiểm `tools/list`, trả lỗi về cho model.
   - Duyệt bằng diff.
3. **MCP thêm:** lọc tool, kinds, prompts, resources, ảnh, `list_changed`, tự khởi động lại, công tắc cho từng tool trên UI.
4. **Provider:**
   - Adapter Anthropic.
   - `api.provider`.
   - Bảng năng lực, chế độ `lean` và `tool_search`.
5. **Tầng 2 do model viết:** `extend` với `inprocess`, luôn hỏi bạn, nạp nóng.
6. **UI "Mở rộng":** một bảng chung cho MCP, extension và plugin, gồm trạng thái, tin cậy, công tắc và lỗi.

## 6. Cần bạn chốt

1. **Model tự viết tool:** chạy trong sandbox (tầng 1) làm mặc định?
2. **Duyệt khi tạo extension:**
   - Tầng 1: hỏi một lần khi tạo hoặc sửa, rồi tự chạy theo kiểu tool.
   - Tầng 2: luôn hỏi.
3. **Đổi `session` thô sang `api`:** plugin `register(api)` hiện tại sẽ phải sửa. Có giữ một lớp tương thích tạm cho chúng không? Đề xuất: không giữ, vì hiện chưa có plugin ngoài nào.
4. **Adapter Anthropic native** có cần ngay không, hay giao thức kiểu OpenAI là đủ?
