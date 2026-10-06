Tôi đã khảo sát cấu trúc dự án (129 file Python, ~54.7k dòng; frontend ~6.1k dòng JS + template) và chia toàn bộ vùng code thành 8 task độc lập, sau đó spawn song song 8 agent con chuyên sâu, mỗi agent phụ trách một vùng:

1. **ash** — app/*.py lõi (main, security, secret_store, cloud, config, upload_utils, schemas, manifest_utils, …)
2. **birch** — app/routers/ (toàn bộ 13 router) + đối chiếu với api.js
3. **cedar** — pipeline (pipeline.py, pipeline_editing.py, optimized_pipeline, page_processing*, mask_store, editorial_gate, parameters, ai_providers)
4. **elm** — app/ocr/ (16 file)
5. **fern** — app/render/, detector/, inpaint/, translation/, visual_qc/
6. **hazel** — app/ai_mode/ + app/downloader/ (SSRF, eval trong generic_js, HTTP timeout…)
7. **ivy** — app/agent/ (sandbox, landlock, session, 22 file)
8. **juniper** — toàn bộ frontend app/static/js + templates + CSS, đối chiếu từng gọi API với routers

**Chưa hoàn thành:** các agent con đã chạy xong (nhận được notification completed) nhưng 6/8 agent trả về "finished without a report" hoặc chỉ trả về một câu trung gian (birch: "Let me verify the security validators…", fern: "Now the inpaint and translation files:", hazel: "Now let me read the job.py in full:") — tức là kết quả audit chi tiết (danh sách phát hiện kèm file:dòng, bằng chứng, cách sửa) **chưa được thu về**. Báo cáo tổng hợp xếp theo mức độ vì thế chưa thể viết.

**Bước tiếp theo tôi sẽ làm:** (1) gọi `wait_agent`/hỏi lại từng agent (đặc biệt ash, cedar, elm, ivy trả về rỗng; birch, fern, hazel bị cắt giữa chừng) để lấy lại danh sách phát hiện đầy đủ, hoặc spawn lại các agent đó với chỉ thị rõ ràng hơn là phải ghi phát hiện vào một file trong temp rồi trả đường dẫn; (2) tự đọc chéo các file trọng yếu (security.py, secret_store.py, routers, api.js) để kiểm chứng chéo từng phát hiện trước khi ghi nhận — nguyên tắc "chỉ nêu những gì đã đọc thấy, không đoán"; (3) gộp, khử trùng lặp (một lỗi có thể bị hai agent cùng chỉ ra), và tổng hợp báo cáo cuối xếp theo mức độ critical → high → medium → low, mỗi mục gồm mức độ, file:dòng, mô tả, bằng chứng trích code và cách sửa.