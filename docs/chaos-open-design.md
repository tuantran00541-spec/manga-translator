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

Quyết định chính: tầng 1 **chính là `create`**, giữ nguyên cách dùng, chỉ mở rộng thêm.

### Tầng 1: `create` (model viết thoải mái)

**Đang có:**
- Model viết một hàm Python, rồi gọi nó như một tool.
- Bên trong, `tools.<tên>(...)` gọi được mọi tool khác: có sẵn, đã create, plugin, MCP. Nhờ vậy một lần gọi chạy được cả chuỗi bước.
- Chạy trong sandbox, duyệt như một lệnh.
- Tool được lưu ở `~/.manga-agent/tools/` để dùng ở các phiên sau.
- Có kiểu một lần (`keep: false`).

**Thêm:**

| Mục | Cách làm |
|---|---|
| `kind` | `create(..., kind="read")` khai báo tool chỉ đọc. Harness kiểm lại: tool `read` mà gọi tool sửa hoặc lệnh thì bị chặn, nên khai `read` không lách được duyệt. Mặc định là `exec`, như hiện nay. |
| `scope` | `user` (mặc định, như hiện nay) hoặc `workspace`. Workspace lưu ở `.agents/tools/`, commit chung với dự án. Người khác mở dự án thì phải tin cậy theo digest, giống `.mcp.json`. |
| Thư viện | Tool đã create gọi lẫn nhau như module (đang có). Cài thêm gói thì model tự chạy `pip install` qua `run_command` (có duyệt); chưa làm `requires` riêng. |
| Danh sách | `/tools` liệt kê, xem code, xóa. Giao diện có bảng tương ứng. |
| Xuất ra MCP | `/tools export <tên>` gói tool thành MCP server, để Claude Code hoặc Cursor dùng lại. |

**Duyệt:** như hiện nay, chạy tool thì duyệt theo `kind`. Tạo hoặc sửa tool không hỏi, vì code chạy trong sandbox; riêng tool workspace của người khác thì phải tin cậy trước.

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

**`ctx` mà handler nhận.** `ctx` là API ổn định chứ không phải sandbox, vì code trong tiến trình vẫn với tới được mọi thứ; ranh giới an toàn thật là bước tin cậy theo digest. `ctx` chỉ gồm những thứ đọc được:
- workspace root;
- `ctx.call(tool, args)`: gọi bất kỳ tool nào như model gọi, qua đúng bước duyệt và hook;
- `ctx.run(cmd)`: chạy lệnh qua sandbox và qua duyệt;
- `ctx.read(path)`;
- `ctx.ask_user(q)`;
- `ctx.emit(kind, data)`;
- cho vòng lặp riêng: `ctx.turn()`, `ctx.run_calls(calls)`, `ctx.append(msg)`, `ctx.default_loop(n)`.

`ctx` không có key, không có `mode` và không có `registry`.

**Model viết plugin tầng 2:**
- **Được viết:** qua tool `plugin_write`, tách riêng khỏi `create` để không lẫn hai mức quyền.
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
| Một giao thức | Chỉ dùng chat completions kiểu OpenAI, vì gần như mọi model và server đều hỗ trợ: OpenAI, Gemini, Claude (lớp tương thích), Ollama, LM Studio, vLLM, OpenRouter, DeepSeek... Không viết adapter riêng cho từng hãng. |
| Tự dò đường dẫn | Bạn dán URL nào cũng được: host trơn, `/v1`, hay endpoint đầy đủ. Chat trả 404 thì thử lần lượt `{base}/chat/completions`, `/v1/chat/completions`, `/api/v1/...`, `/openai/v1/...`, `/v1beta/openai/...`, rồi nhớ đường nào chạy được. Hiện mới dò khi thêm provider và chỉ qua `/models`, nên server không có `/models` sẽ thiếu. |
| Dự phòng cho giao thức lạ | `api.provider(name, adapter)` trong plugin, với `adapter(request) -> {"text", "calls", "usage"}` theo đúng định dạng nội bộ mà `client.complete` trả. Chỉ dùng khi gặp model không nói được chat completions. |
| Bảng năng lực model | `profile.json` → `models.<id>` khai báo: `tools` (native/text), `vision`, `reasoning`, `context_window`, `max_output`. Mục nào thiếu thì đoán từ `models.py`, và đoán sai thì tự sửa theo lỗi lần đầu (đã làm cho context window). |
| Model yếu | Đặt `"tools": "lean"` thì chỉ gửi tool cốt lõi (read, edit, run, search). Tool MCP và tool đã create chỉ được gửi khi model gọi `tool_search` (tìm theo tên hoặc mô tả). |
| Prompt trung lập | System prompt không giả định hãng. Hướng dẫn riêng cho từng model đặt ở `models.<id>.prompt` hoặc trong plugin. |

## 5. Thứ tự làm

Mỗi bước có test, 4 script sanity và ảnh chụp UI, rồi mới sang bước sau.

1. **API plugin v1:**
   - Đối tượng `api`/`ctx` thay cho `session` thô.
   - Thêm 4 sự kiện mới.
   - `api.config` và `api.state`.
   - Test bảo mật: plugin không với tới key, mode, registry.
2. **Mở rộng `create`:** `kind` có kiểm tra, `scope` workspace kèm tin cậy, lệnh `/tools`, xuất ra MCP.
3. **MCP thêm:** lọc tool, kinds, prompts, resources, ảnh, `list_changed`, tự khởi động lại, công tắc cho từng tool trên UI.
4. **Provider:**
   - Dò đường dẫn chat khi gặp 404, và nhớ đường chạy được.
   - `api.provider`.
   - Bảng năng lực, chế độ `lean` và `tool_search`.
5. **Tầng 2 do model viết:** `plugin_write`, luôn hỏi bạn, nạp nóng.
6. **UI "Mở rộng":** một bảng chung cho MCP, tool đã create và plugin, gồm trạng thái, tin cậy, công tắc và lỗi.

## 6. Đã chốt

1. Tầng 1 là `create`, mở rộng như bảng ở mục 3.
2. **Duyệt:**
   - `create`: không hỏi khi tạo hoặc sửa tool.
   - `plugin_write`: luôn hỏi, vì plugin model viết chạy trong harness và chưa rõ nó sẽ kết nối tới đâu.
3. Đổi sang bộ API mới và không giữ lớp tương thích.
4. Chỉ dùng một giao thức chat completions kiểu OpenAI, kèm tự dò đường dẫn. Giao thức khác thì đi qua plugin.
