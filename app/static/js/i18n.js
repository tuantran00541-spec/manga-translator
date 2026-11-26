(() => {
  "use strict";

  const STORAGE_KEY = "manga_ui_lang";
  const DEFAULT_LANG = "vi";
  const SUPPORTED = new Set(["vi", "en"]);
  const ATTRS = ["placeholder", "title", "aria-label"];

  const viRegex = /[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]/;

  const EXACT = Object.assign(Object.create(null), {
    // Header & Navigation
    "Mở thanh điều hướng": "Open navigation sidebar",
    "Đóng thanh điều hướng": "Close navigation sidebar",
    "Điều hướng chính": "Main navigation",
    "Các màn hình": "Screens",
    "Manga Translator — về trang chủ": "Manga Translator — go home",
    "Không gian làm việc": "Workspace",
    "Trang chủ": "Home",
    "Nhập nội dung": "Import",
    "Tải video": "Video Downloader",
    "Tiện ích đa phương tiện": "Media Utility",
    "Tải video & âm thanh": "Video & Audio Downloader",
    "Dán liên kết TikTok, YouTube hoặc Facebook. Ứng dụng sẽ kiểm tra độ an toàn của liên kết trước khi hiển thị tùy chọn tải 1080p, 720p hoặc chỉ xuất tệp âm thanh.": "Paste a TikTok, YouTube, or Facebook link. The app verifies link safety before showing 1080p, 720p, or audio-only download options.",
    "Dán liên kết video": "Paste video link",
    "Kiểm tra giao thức HTTPS, tên miền chính chủ, chống SSRF và chuỗi chuyển hướng trước khi tải.": "Verifies HTTPS protocol, official platform domain, SSRF protection, and redirect chain before downloading.",
    "Nền tảng hỗ trợ": "Supported platforms",
    "Liên kết TikTok, YouTube hoặc Facebook": "TikTok, YouTube, or Facebook URL",
    "https://www.youtube.com/watch?v=... hoặc link TikTok / Facebook": "https://www.youtube.com/watch?v=... or TikTok / Facebook link",
    "Dán": "Paste",
    "Kiểm tra liên kết": "Check link",
    "Đang kiểm tra…": "Checking…",
    "Đang kiểm tra": "Checking",
    "Liên kết an toàn": "Safe link",
    "Cảnh báo — Đã chặn": "Warning — Blocked",
    "Chọn chất lượng tải xuống": "Choose download quality",
    "Tùy chọn chất lượng video và âm thanh": "Video and audio quality options",
    "Tải xuống ngay": "Download now",
    "Đang khởi tạo…": "Starting…",
    "Kiểm tra link khác": "Check another link",
    "Lưu về máy": "Save to device",
    "Hủy tải": "Cancel download",
    "Tệp đã tải gần đây": "Recent downloads",
    "Tệp tạm được giữ tự động trong 2 giờ để bạn tải lại hoặc xóa gọn.": "Temporary files are kept for 2 hours so you can re-download or clean them up.",
    "Lưu tệp": "Save file",
    "Chỉ xuất Audio (MP3)": "Audio only (MP3)",
    "Chỉ xuất Audio (M4A)": "Audio only (M4A)",
    "Đang chuẩn bị luồng tải…": "Preparing download stream…",
    "Đang tải xuống dữ liệu…": "Downloading media…",
    "Đang ghép hình ảnh & âm thanh…": "Merging video & audio…",
    "Đang hủy tác vụ…": "Canceling task…",
    "Hoàn tất! Tệp đã sẵn sàng lưu về máy.": "Complete! Ready to save to your device.",
    "Đã hủy tải xuống.": "Download canceled.",
    "Tải xuống thất bại.": "Download failed.",
    "Xem cắt lát": "Slice Preview",
    "Xử lý & Biên tập": "Process & Edit",
    "Cài đặt": "Settings",
    "Chưa mở chương": "No chapter open",
    "Dự án mới": "New project",
    "Giao diện màu": "Color theme",
    "Giao diện": "Theme",
    "Hệ thống": "System",
    "Sáng": "Light",
    "Tối": "Dark",

    // Home
    "Xin chào!": "Welcome!",
    "Chọn một chương gần đây hoặc bắt đầu dự án dịch mới.": "Pick a recent chapter or start a new translation project.",
    "Chương đang xử lý": "Active chapters",
    "Dự án gần đây": "Recent projects",
    "Chưa có chương gần đây": "No recent chapters yet",
    "Nhập liên kết, ảnh hoặc tệp truyện để bắt đầu.": "Enter a URL, images, or comic archive to get started.",
    "Bắt đầu dự án đầu tiên": "Start first project",
    "Chương chưa đặt tên": "Untitled chapter",
    "Không có ảnh bìa": "No cover image",

    // Import & AI Mode
    "Dán liên kết chương hoặc tải ảnh, ZIP, CBZ từ máy. Ngôn ngữ gốc được tự nhận diện sau khi xử lý ảnh.": "Paste a chapter URL or upload images, ZIP, or CBZ files. Source language is auto-detected after processing.",
    "Mở chương mới": "Open new chapter",
    "Chế độ AI": "AI Mode",
    "Chương Chế độ AI": "AI Mode Chapter",
    "Dán liên kết chương, AI tự động xử lý toàn bộ: loại bỏ trang giới thiệu, giữ lại biểu trưng, làm sạch và vá nền vùng còn sót, dịch thuật, chọn phông chữ và kết xuất. Sau khi hoàn tất, chương đã chèn chữ sẽ mở sẵn để bạn kiểm tra, chỉnh sửa và xuất bản.": "Paste a chapter URL and AI handles everything automatically: filtering credit pages, preserving logos, cleaning and inpainting residue, translating, picking fonts, and rendering. When finished, the typeset chapter opens ready for review, editing, and export.",
    "Chế độ chạy AI": "AI run mode",
    "1 chương": "Single chapter",
    "Nhiều chương (Hàng đợi)": "Multiple chapters (Queue)",
    "Liên kết chương cho Chế độ AI": "Chapter URL for AI Mode",
    "Dán URL chương tại đây": "Paste chapter URL here",
    "Liên kết mục lục truyện": "Series table of contents URL",
    "Dán URL mục lục truyện để tự quét danh sách chương (tùy chọn)": "Paste series page URL to scan chapters automatically (optional)",
    "Quét chương": "Scan chapters",
    "Từ": "From",
    "đến": "to",
    "Lấy khoảng": "Apply range",
    "Lô 30 tiếp": "Next 30",
    "Chuyển sang lô 30 chương kế tiếp": "Jump to the next 30 chapters",
    "Tất cả": "All",
    "Bỏ qua chương lỗi để chạy tiếp chương sau (lỗi tải trang, xử lý ảnh hoặc dịch từng chương)": "Skip failed chapters and continue queue (download, image processing, or per-chapter translation errors)",
    "Nếu một chương bị lỗi tải liên kết (404/Cloudflare), lỗi tách chữ hoặc lỗi dịch riêng lẻ, hàng đợi sẽ ghi nhận lỗi đó và tự động chuyển sang chương kế tiếp. Lỗi khóa API hoặc hết số dư Manga Cloud vẫn dừng hàng đợi.": "If one chapter fails to fetch (404/Cloudflare) or errors during detection/translation, the queue logs it and moves to the next chapter. API key or empty Manga Cloud balance errors still pause the queue.",
    "Nhà cung cấp AI": "AI Provider",
    "Mô hình thị giác": "Vision model",
    "Mặc định": "Default",
    "Dùng mô hình thị giác mặc định trong Cài đặt": "Use default vision model from Settings",
    "Theo thiết lập của nhà cung cấp": "Provider default",
    "Tìm hoặc nhập ID mô hình…": "Search or type a model ID…",
    "Nhập mô hình tùy chỉnh": "Custom model ID",
    "Đang tải danh sách mô hình…": "Loading model list…",
    "Không tìm thấy mô hình phù hợp.": "No matching models found.",
    "Dịch sang": "Translate to",
    "Tiếng Việt": "Vietnamese",
    "Số ảnh mỗi lần gửi": "Images per request",
    "Số lượng lát ảnh gửi đồng thời trong một lượt quét và trích xuất thuật ngữ. Nếu mô hình hỗ trợ nhiều ảnh, có thể tăng giá trị này để xử lý nhanh và tiết kiệm chi phí hơn.": "Number of slices sent together during scan and glossary extraction. Increase if your model supports multi-image batches for faster, cheaper runs.",
    "Giới hạn chi phí (USD)": "Cost budget (USD)",
    "Bối cảnh truyện (tùy chọn)": "Story context (optional)",
    "Thể loại, nhân vật chính, cách xưng hô. Ví dụ: Hồi quy học viện. Ian (17 tuổi) xưng tôi, gọi thầy Baldur là thầy.": "Genre, main characters, tone/pronouns. E.g.: Academy regression. Ian speaks casually; characters address Master Baldur respectfully.",
    "Bộ nhớ bộ truyện": "Series Memory",
    "Tên riêng, thuật ngữ, cách xưng hô và nhân vật đã thống nhất từ các chương trước để áp dụng đồng bộ cho các chương sau. Mỗi dòng một mục:": "Names, glossary terms, address rules, and character notes synced across chapters. One entry per line:",
    "Tên gốc = Tên dịch": "Source = Translation",
    "term: từ gốc = bản dịch": "term: source = translation",
    "A -> B: tôi / anh": "A -> B: I / you",
    "char: Tên | ghi chú": "char: Name | note",
    "Dán liên kết chương rồi mở mục này để xem bộ nhớ của bộ truyện.": "Paste a chapter URL then open this section to inspect series memory.",
    "Bộ truyện này chưa có chương nào chạy Chế độ AI; bạn có thể nhập trước tên riêng và cách xưng hô.": "This series has no prior AI Mode chapters; you can pre-enter names and address rules here.",
    "Đã lưu bộ nhớ bộ truyện.": "Series memory saved.",
    "Lưu bộ nhớ": "Save memory",
    "Nâng cao: hiệu đính câu dịch. Các câu chưa tự nhiên, thiếu mạch lạc hoặc tối nghĩa sẽ được gửi lại để AI biên tập.": "Advanced: proofread translated lines. Unnatural, awkward, or unclear lines are sent back for AI refinement.",
    "Chạy Chế độ AI": "Run AI Mode",
    "Chạy hàng đợi": "Run queue",
    "Thử lại mục lỗi": "Retry failed",
    "Hủy hàng đợi": "Cancel queue",
    "Mở chương": "Open chapter",
    "Hủy": "Cancel",
    "Báo cáo của AI": "AI Report",
    "Email của bạn": "Your email",
    "Mã 6 số": "6-digit code",
    "Gửi mã": "Send code",
    "Nạp tiền": "Top up",
    "Tài khoản": "Account",
    "Lịch sử số dư": "Balance history",
    "Chưa có giao dịch nào.": "No transactions yet.",
    "Đăng nhập": "Sign in",
    "Đăng xuất": "Sign out",
    "Đăng xuất mọi thiết bị": "Sign out all devices",
    "Xóa tài khoản": "Delete account",
    "Đã xóa tài khoản": "Account deleted",
    "Đã đăng xuất mọi thiết bị": "Signed out all devices",
    "Số dư còn lại vẫn được bảo lưu và khôi phục khi bạn đăng nhập lại bằng đúng email này.": "Remaining balance stays preserved and restores when you sign back in with this email.",
    "Đăng nhập để sử dụng Manga Cloud, hoặc chọn nhà cung cấp khác với khóa API cá nhân (miễn phí).": "Sign in to use Manga Cloud, or choose another provider with your own API key (free).",
    "Không thể kết nối Manga Cloud — vẫn có thể sử dụng khóa API cá nhân.": "Could not reach Manga Cloud — personal API keys still work.",
    "Đăng xuất Manga Cloud trên mọi máy đang dùng tài khoản này?": "Sign out Manga Cloud on all devices using this account?",
    "Xóa tài khoản Manga Cloud? Địa chỉ email và phiên đăng nhập sẽ bị xóa.": "Delete Manga Cloud account? Your email and session will be removed.",
    "Phiên đăng nhập đã hết hạn — vui lòng đăng nhập lại.": "Session expired — please sign in again.",
    "QR ngân hàng": "Bank QR",
    "Thẻ quốc tế": "International card",
    "Hoàn tiền": "Refund",
    "Điều chỉnh": "Adjustment",
    "Đang chờ xác nhận thanh toán…": "Waiting for payment confirmation…",
    "Đã mở trang thanh toán. Số dư tự cộng sau khi thanh toán xong.": "Checkout opened. Balance updates automatically after payment.",
    "Chế độ AI đã hoàn tất, đang mở chương.": "AI Mode finished, opening chapter.",
    "AI đang xử lý, bạn có thể giữ trang này mở hoặc quay lại sau.": "AI is processing; you can keep this page open or check back later.",
    "Hoàn tất! Chương đã được chèn chữ và sẵn sàng để kiểm tra, chỉnh sửa và xuất bản.": "Complete! Chapter is typeset and ready for review, editing, and export.",
    "Đang hủy hàng đợi sau chương hiện tại…": "Canceling queue after current chapter…",
    "Đang hủy sau bước hiện tại…": "Canceling after current step…",
    "Đã hủy": "Canceled",
    "Đã hủy. Kết quả đã thực hiện vẫn được lưu trong chương.": "Canceled. Completed work remains saved in the chapter.",
    "Dừng do chương trước gặp lỗi": "Stopped due to prior chapter error",
    "Xong (có lỗi)": "Done (with errors)",
    "Chờ": "Pending",
    "Cần rà": "Needs review",
    "Lỗi": "Error",
    "dịch thuật": "translation",
    "hiệu đính câu": "line polish",
    "kết xuất": "rendering",
    "mặc định": "default",
    "quét trang giới thiệu/biểu trưng": "credit/logo scan",
    "đối chiếu ảnh gốc và nền sạch": "original vs clean check",
    "Vui lòng dán liên kết chương trước.": "Please paste a chapter URL first.",
    "Vui lòng dán liên kết trang mục lục truyện trước.": "Please paste a series table-of-contents URL first.",
    "Vui lòng dán ít nhất 1 liên kết chương.": "Please paste at least 1 chapter URL.",
    "Không tìm thấy liên kết chương nào từ trang này.": "No chapter links found on this page.",

    // URL & File Upload + Extension Guide
    "Mở từ liên kết": "Open from URL",
    "Hỗ trợ liên kết chương từ nguồn truyện.": "Supports chapter links from comic sites.",
    "Liên kết chương": "Chapter URL",
    "Tải chương": "Load chapter",
    "Trang chặn tải tự động (Cloudflare)?": "Blocked by anti-bot protection (Cloudflare)?",
    "Dùng tiện ích trình duyệt": "Use browser extension",
    "Tải tệp lên": "Upload files",
    "PNG, JPG, WEBP, BMP, ZIP hoặc CBZ.": "PNG, JPG, WEBP, BMP, ZIP, or CBZ.",
    "Kéo thả tệp vào đây": "Drag & drop files here",
    "hoặc": "or",
    "chọn từ thiết bị": "choose from device",
    "Có thể chọn nhiều ảnh cùng lúc": "Multiple images can be selected",
    "Cài đặt tiện ích gửi chương": "Install Chapter Sender Extension",
    "Tiện ích giúp thu thập ảnh trực tiếp từ chương bạn đang mở trên Chrome hoặc Edge, vượt qua cơ chế chặn tải tự động. Chỉ cần cài đặt một lần:": "The extension captures images directly from your open chapter tab in Chrome or Edge, bypassing anti-bot blocks. One-time setup:",
    "Mở trang Tiện ích: nhập": "Open the Extensions page: enter",
    "(hoặc": "(or",
    "trên Edge) vào thanh địa chỉ.": "in Edge) into the address bar.",
    "Sao chép": "Copy",
    "Đã sao chép.": "Copied.",
    "Đã sao chép": "Copied",
    "Không thể sao chép tự động, vui lòng chọn và sao chép thủ công.": "Could not copy automatically; please select and copy manually.",
    "Bật": "Enable",
    "Chế độ dành cho nhà phát triển": "Developer mode",
    "(Developer mode) ở góc trên bên phải.": "at the top-right corner.",
    "/500, mỗi dòng một URL)": "/500, one URL per line)",
    "Danh sách liên kết chương (": "Chapter link list (",
    "https://example.com/truyen/chuong-1\nhttps://example.com/truyen/chuong-2": "https://example.com/series/chapter-1\nhttps://example.com/series/chapter-2",
    "ở góc trên bên phải.": "at the top-right corner.",
    "Tải tệp tiện ích": "Download extension",
    "bên dưới rồi giải nén vào một thư mục dễ tìm (ví dụ": "below and extract it to an easy-to-find folder (e.g.",
    "Tải tiện ích đã giải nén": "Load unpacked",
    "(Load unpacked) và chọn đúng thư mục vừa giải nén.": "and choose the extracted folder.",
    "Tải tiện ích (.zip)": "Download extension (.zip)",
    "Cách sử dụng: mở chương truyện, nhấp vào biểu tượng Manga Translator trên thanh tiện ích rồi chọn": "Usage: open any comic chapter, click the Manga Translator icon on the extension bar, and choose",
    "Gửi chương vào app": "Send chapter to app",
    ". Nếu Chrome hiển thị thông báo tắt tiện ích ở chế độ nhà phát triển, vui lòng chọn giữ lại.": ". If Chrome prompts to keep or disable developer-mode extensions, choose Keep.",
    "Phiên bản tiện ích đang cài đã cũ hơn phiên bản trong ứng dụng: vui lòng tải lại gói": "Your installed extension is older than the app's version: please download the new",
    "mới, giải nén đè rồi chọn": "package, extract over the existing folder, and click",
    "Tải lại": "Reload",
    "tại thẻ Manga Translator.": "on the Manga Translator card.",
    "Đóng": "Close",

    // Settings & Proxy Monitor
    "Đang tải cấu hình…": "Loading configuration…",
    "Đóng bảng cài đặt": "Close settings",
    "Danh mục cài đặt": "Settings categories",
    "Chung": "General",
    "Mô hình AI": "AI Models",
    "Proxy API": "API Proxy",
    "Ngôn ngữ ứng dụng": "App language",
    "Chọn ngôn ngữ hiển thị toàn bộ giao diện ứng dụng": "Choose display language for the entire application UI",
    "Hiệu năng và mức sử dụng": "Performance & telemetry",
    "Mức độ hiển thị thông tin token, bộ nhớ đệm và thời gian thực thi": "Detail level for token, cache, and latency metrics",
    "Chi tiết": "Detailed",
    "Gọn": "Compact",
    "Tự động thu gọn bước chạy": "Auto-collapse steps",
    "Chọn thời điểm tự đóng chi tiết công cụ trong Chaos": "When to automatically collapse tool step groups in Chaos",
    "Khi hoàn tất": "When done",
    "Không bao giờ": "Never",
    "Quyền hạn thực thi": "Execution permissions",
    "Chọn chế độ phê duyệt mặc định cho phiên làm việc mới": "Default approval mode for new sessions",
    "Phạm vi truy cập tệp (Sandbox)": "Filesystem sandbox scope",
    "Giới hạn thư mục mà các lệnh được phép ghi": "Restrict which folders commands can modify",
    "Nhà cung cấp AI mặc định": "Default AI provider",
    "+ Thêm nhà cung cấp": "+ Add provider",
    "Kết nối dịch vụ trung gian, máy chủ tự lưu trữ hoặc điểm cuối tương thích OpenAI / Gemini thông qua URL gốc, giao thức và danh sách mô hình.": "Connect relay gateways, self-hosted servers, or OpenAI/Gemini-compatible endpoints via Base URL, protocol, and model list.",
    "Mã định danh nhà cung cấp": "Provider ID",
    "Định danh viết thường (a-z, 0-9, -), dùng để phân biệt nhà cung cấp trong yêu cầu và kho khóa bảo mật.": "Lowercase identifier (a-z, 0-9, -) used to distinguish this provider in requests and secret storage.",
    "Mã định danh nhà cung cấp chỉ gồm chữ thường, số, dấu gạch ngang hoặc gạch dưới.": "Provider ID may only contain lowercase letters, digits, hyphens, or underscores.",
    "Tên hiển thị": "Display name",
    "Tên hiển thị (VD: MiniRouter)": "Display name (e.g. MiniRouter)",
    "Giao thức API": "API protocol",
    "URL gốc (Base URL)": "Base URL",
    "Khóa API": "API Key",
    "Nhập khóa API của bạn": "Enter your API key",
    "Đã lưu khóa — nhập khóa mới nếu muốn thay thế hoặc bổ sung": "Key saved — enter a new key to replace or append",
    "Giữ lại các khóa đã lưu và bổ sung thêm khóa này": "Keep existing saved keys and append this key",
    "Lưu nhà cung cấp": "Save provider",
    "Thêm khóa": "Add key",
    "Xóa nhà cung cấp": "Remove provider",
    "Danh sách mô hình": "Model list",
    "Tải danh sách mô hình": "Fetch model list",
    "Tìm nhanh mô hình…": "Quick filter models…",
    "Tìm mô hình trong danh sách": "Filter models in list",
    "Chọn mặc định": "Set default",
    "Đang chọn": "Selected",
    "Chưa có mô hình nào được tải. Lưu nhà cung cấp hoặc chọn 'Tải danh sách mô hình' để lấy danh sách từ máy chủ.": "No models loaded yet. Save provider or click 'Fetch model list' to load from upstream.",
    "Chưa có khóa": "No key",
    "Đã kết nối": "Connected",
    "Đang kiểm tra cấu hình…": "Checking configuration…",
    "Đang tải…": "Loading…",
    "Chưa có nhà cung cấp nào": "No providers configured",
    "Không đọc được cấu hình AI": "Could not load AI settings",
    "Hãy chọn hoặc lưu nhà cung cấp trước.": "Please select or save a provider first.",
    "Vui lòng nhập Tên hiển thị, URL gốc và Khóa API.": "Please enter Display name, Base URL, and API Key.",
    "Loại đầu vào hỗ trợ:": "Supported inputs:",
    "Hình ảnh": "Images",
    "Văn bản": "Text",
    "Mức độ suy luận:": "Reasoning effort:",
    "Thấp (low)": "Low",
    "Vừa (medium)": "Medium",
    "Cao (high)": "High",
    "Tối đa (xhigh)": "Maximum (xhigh)",

    // Proxy Monitor
    "Proxy cứu lỗi API": "API Error-Rescue Proxy",
    "Tự động bật · bỏ qua 429/quota": "Auto-enabled · skips 429/quota",
    "Làm mới": "Refresh",
    "Xóa log": "Clear logs",
    "Endpoint nội bộ:": "Local endpoint:",
    "Đã sao chép địa chỉ Proxy API.": "Proxy API URL copied.",
    "Tổng gọi": "Total calls",
    "Đã cứu": "Rescued",
    "Bỏ qua 429": "Skipped 429",
    "Bỏ qua 429/Quota": "Skipped 429/Quota",
    "Lỗi khác": "Other errors",
    "Chưa ghi nhận lỗi hay lượt cứu nào trong phiên chạy máy chủ này.": "No errors or rescues recorded in this server session yet.",
    "Gỡ bọc": "Unwrap",
    "Dịch ảnh": "Image translation",

    // Chaos Agent
    "Phiên": "Sessions",
    "Phiên mới": "New session",
    "Phiên làm việc": "Session",
    "Mở danh sách phiên": "Open sessions sidebar",
    "Đóng danh sách phiên": "Close sessions sidebar",
    "Tùy chỉnh": "Customize",
    "Tùy chỉnh Chaos": "Customize Chaos",
    "Kết nối các kỹ năng, máy chủ MCP và thiết lập môi trường để Chaos làm việc trên dự án của bạn.": "Connect skills, MCP servers, and environment settings for Chaos on your workspace.",
    "Quay lại trò chuyện": "Back to chat",
    "Danh mục tùy chỉnh": "Customization tabs",
    "Tìm phiên làm việc": "Search sessions",
    "Tìm phiên làm việc…": "Search sessions…",
    "Chưa có phiên nào.": "No sessions yet.",
    "Không tìm thấy phiên phù hợp.": "No matching sessions found.",
    "Kỹ năng": "Skills",
    "Thiết lập": "Setup",
    "Nạp kỹ năng mới": "Install new skill",
    "Nhập CHỦ/REPO hoặc CHỦ/REPO/thư-mục trên GitHub…": "Enter OWNER/REPO or OWNER/REPO/folder on GitHub…",
    "Tải lên .zip / SKILL.md": "Upload .zip / SKILL.md",
    "Đã cài đặt": "Installed",
    "Chưa có kỹ năng nào.": "No skills yet.",
    "Tạo hoặc chỉnh sửa kỹ năng thủ công": "Create or edit skill manually",
    "Tên (a-z, 0-9, -)": "Name (a-z, 0-9, -)",
    "Mô tả: chức năng và ngữ cảnh áp dụng": "Description: what it does and when to use",
    "Hướng dẫn (Markdown)": "Instructions (Markdown)",
    "Lưu kỹ năng": "Save skill",
    "Nhờ trợ lý soạn thảo": "Ask assistant to draft",
    "Máy chủ MCP hiện có": "Configured MCP servers",
    "Chưa có máy chủ MCP nào.": "No MCP servers configured.",
    "Thêm máy chủ MCP": "Add MCP server",
    "Phương thức chạy máy chủ": "Server transport",
    "Lệnh": "Command",
    "Tên": "Name",
    "Lệnh khởi chạy máy chủ": "Server startup command",
    "Biến môi trường (TÊN=giá trị, mỗi dòng một biến)": "Environment variables (NAME=value, one per line)",
    "URL máy chủ": "Server URL",
    "Tiêu đề HTTP (Tên: giá trị, mỗi dòng một mục)": "HTTP Headers (Name: value, one per line)",
    "Cấu hình JSON (\"mcpServers\": {…} hoặc một máy chủ)": "JSON config (\"mcpServers\": {…} or single server)",
    "Thêm và kiểm tra kết nối": "Add & test connection",
    "Môi trường & Cơ chế cô lập (Sandbox)": "Environment & Sandbox",
    "Thư mục làm việc": "Working directory",
    "Môi trường thực thi lệnh": "Command execution runtime",
    "Cô lập ở cấp hệ điều hành": "OS-level sandbox",
    "Ảnh Docker (Image)": "Docker image",
    "Dữ liệu & Công cụ phiên hiện tại": "Current session context & tools",
    "Tiến độ kế hoạch": "Plan progress",
    "Tin nhắn": "Message",
    "Việc cần làm… (/ xem lệnh)": "Task to do… (/ for commands)",
    "Quyền": "Permissions",
    "Phê duyệt": "Approval",
    "Hỏi trước": "Ask first",
    "Tự sửa và chạy": "Auto-edit & run",
    "Tự động (rủi ro)": "Auto (full trust)",
    "Cô lập (Sandbox)": "Sandbox",
    "Chỉ đọc": "Read-only",
    "Ghi trong thư mục": "Workspace write",
    "Không giới hạn cô lập": "Full access",
    "Cho phép lệnh truy cập mạng": "Allow network access",
    "Nhà cung cấp và mô hình": "Provider & model",
    "Nhà cung cấp": "Provider",
    "Mô hình": "Model",
    "Chọn mô hình": "Choose model",
    "Tìm mô hình…": "Search models…",
    "Mức độ suy luận": "Reasoning effort",
    "Mặc định của mô hình": "Model default",
    "Theo thiết lập gốc của mô hình": "Model's native default",
    "Thấp": "Low",
    "Vừa": "Medium",
    "Khuyên dùng": "Recommended",
    "Cao": "High",
    "Tối đa": "Maximum",
    "Suy luận": "Reasoning",
    "Suy luận càng sâu, câu trả lời càng kỹ nhưng thời gian chờ và mức tiêu hao sẽ cao hơn.": "Deeper reasoning yields thorough answers with higher latency and token cost.",
    "Thêm mô hình": "More models",
    "Chưa chọn": "Not selected",
    "Chưa có mô hình": "No models",
    "Chưa có nhà cung cấp": "No providers",
    "Chưa có nhà cung cấp nào được cấu hình.": "No configured providers.",
    "Chưa có nhà cung cấp nào. Thêm trong Cài đặt, mục Nhà cung cấp AI.": "No providers configured. Add one in Settings > AI Models.",
    "Không có mô hình": "No models available",
    "Không tải được mô hình": "Failed to load models",
    "Dừng": "Stop",
    "Gửi": "Send",
    "Sẵn sàng": "Ready",
    "Đang chạy": "Running",
    "Chờ duyệt": "Waiting approval",
    "Chế độ lập kế hoạch (/plan)": "Plan mode (/plan)",
    "Công cụ văn bản": "Text tools",
    "Tự sửa": "Auto-edit",
    "Tự động": "Auto",
    "chỉ đọc": "read-only",
    "ghi thư mục": "workspace-write",
    "không cô lập": "full-access",
    "Quá trình suy luận": "Reasoning trace",
    "Cập nhật kế hoạch": "Update plan",
    "Kế hoạch chờ duyệt": "Plan awaiting approval",
    "Duyệt": "Approve",
    "Duyệt, tự động": "Approve & auto",
    "Sửa lại": "Revise",
    "Cần sửa gì?": "What needs changes?",
    "Cho phép": "Allow",
    "Từ chối": "Deny",
    "Bỏ qua": "Skip",
    "Câu trả lời của bạn": "Your answer",
    "Lý do từ chối (tùy chọn)": "Rejection reason (optional)",
    "Đã cho phép.": "Allowed.",
    "Đã chuyển sang tự sửa và chạy.": "Switched to auto-edit & run.",
    "Đã từ chối.": "Denied.",
    "Đã bỏ qua câu hỏi.": "Skipped question.",
    "Đã yêu cầu sửa.": "Revision requested.",
    "Đã xếp hàng, đọc ở bước sau": "Queued for next step",
    "Vui lòng chọn nhà cung cấp và mô hình.": "Please choose a provider and model.",
    "Vừa đọc nội dung từ web hoặc MCP, có thể chứa lệnh giả.": "Just read external web/MCP content; verify against prompt injection.",
    "Phần mở rộng chạy với toàn quyền, không qua cơ chế cô lập. Vui lòng kiểm tra kỹ mã nguồn trước khi cho phép.": "Plugins run unsandboxed with full permissions. Review source code before allowing.",
    "Khảo sát & Lập kế hoạch": "Explore & Plan",
    "Đọc kiến trúc & đề xuất từng bước": "Read codebase architecture & propose steps",
    "Kiểm toán & Sửa lỗi": "Audit & Fix",
    "Rà soát lỗi, luồng dữ liệu & đề xuất bản vá": "Review bugs, data flow & propose patches",
    "Mục tiêu tự hành": "Autonomous Goal",
    "Tự động thực thi & kiểm thử đến khi hoàn tất": "Run & verify automatically until complete",
    "Tạo Kỹ năng & Công cụ": "Create Skill & Tool",
    "Mở rộng khả năng chuyên biệt cho Chaos": "Extend Chaos with custom skills",
    "của bạn": "user",
    "của dự án": "project",
    "có sẵn": "built-in",
    "đang chạy": "running",
    "chờ cho phép": "pending trust",
    "chờ": "waiting",
    "đã tắt": "disabled",
    "đã dừng": "stopped",
    "đã nạp": "loaded",
    "chưa cho phép": "untrusted",
    "tắt": "off",
    "lỗi": "error",
    "đang kiểm tra…": "testing…",
    "Sửa": "Edit",
    "Thử": "Test",
    "Tắt": "Disable",
    "Xóa": "Delete",
    "Nhân đôi": "Duplicate",
    "Nhân đôi vùng chữ (Ctrl+D)": "Duplicate text box (Ctrl+D)",
    "Nhân đôi vùng chữ đang chọn": "Duplicate selected text box",
    "Đã nhân đôi vùng chữ.": "Duplicated text box.",
    "Hình dạng vùng": "Region shape",
    "Chữ nhật": "Rectangle",
    "Elip": "Ellipse",
    "Nền khung": "Box fill",
    "Trong suốt": "Transparent",
    "Kem": "Cream",
    "Bo góc": "Corner radius",
    "Màu nền khung tùy chỉnh": "Custom box background color",
    "Chuyển sắc": "Gradient",
    "Dọc": "Vertical",
    "Ngang": "Horizontal",
    "Hào quang": "Glow",
    "Màu hào quang (Glow)": "Glow color",
    "Bật": "On",
    "Hút màu (I)": "Eyedropper (I)",
    "Hút màu từ ảnh": "Sample color from image",
    "Hút màu trực tiếp từ ảnh (I)": "Sample color directly from image (I)",
    "Nhấp lên ảnh để lấy màu chữ. Giữ Shift: màu viền, Alt: nền khung.": "Click image to pick text color. Hold Shift: stroke, Alt: box background.",
    "Nhấp lên ảnh để hút màu chữ (Shift: viền, Alt: nền).": "Click image to sample text color (Shift: stroke, Alt: background).",
    "Giữ để xem ảnh gốc (\\)": "Hold to peek original (\\)",
    "Giữ chuột hoặc phím \\ để đối chiếu ảnh gốc tức thì": "Hold mouse or \\ key to compare with original instantly",
    "Bật/tắt xem trước chữ trên nền sạch (T)": "Toggle live text preview on clean canvas (T)",
    "Xem trước chữ trực tiếp trong vùng chọn (T)": "Live text preview inside regions (T)",
    "Ẩn/hiện khung vùng chữ (Q)": "Show/hide region overlays (Q)",
    "Ẩn / hiện khung vùng chữ (Q)": "Show / hide text overlays (Q)",
    "Hiển thị vùng chữ và đối chiếu": "Overlay display & comparison",
    "Dùng độ dày và màu viền tự động": "Use automatic stroke width and color",
    "Tự động tương phản / Màu gốc OCR": "Auto contrast / OCR source color",
    "Vi chỉnh vị trí vùng đang chọn (1px / 10px)": "Nudge selected box (1px / 10px)",
    "Xem nhanh ảnh gốc để đối chiếu": "Peek original image to compare",
    "Ẩn/hiện khung vùng chữ · Bật/tắt xem trước chữ": "Show/hide overlays · Toggle live text preview",
    "Chọn, chữ nhật, elip, cọ, tẩy, hút màu, bàn tay, thu phóng": "Select, rect, ellipse, brush, eraser, eyedropper, hand, zoom",
    "gọi MCP": "call MCP",
    "Không bật cơ chế cô lập: mọi lệnh đều cần phê duyệt, trừ khi bật chế độ Tự động.": "Sandbox off: commands require approval unless Auto mode is active.",
    "Windows mức toàn vẹn thấp: chỉ được ghi trong thư mục làm việc; thao tác đọc tệp và truy cập mạng vẫn cần phê duyệt.": "Windows low-integrity sandbox: writes limited to workspace; reads outside & network require approval.",
    "Container Docker, chỉ truy cập được /workspace.": "Docker container, restricted to /workspace.",
    "Cơ chế cô lập Landlock (Linux).": "Landlock sandbox (Linux).",
    "Cơ chế cô lập Seatbelt (macOS).": "Seatbelt sandbox (macOS).",
    "Đã tắt cơ chế cô lập: lệnh có toàn quyền thực thi.": "Sandbox disabled: commands have full access.",
    "Không tìm thấy Docker, chuyển sang cơ chế cô lập của hệ điều hành.": "Docker not found; falling back to OS sandbox.",
    "Không rõ thời gian chờ.": "Retry wait unknown.",
    "Đọc": "Read",
    "Ghi": "Write",
    "Chạy": "Run",
    "Tìm": "Search",
    "Tìm tệp": "Find files",
    "Tìm ký hiệu": "Find symbols",
    "Xem thư mục": "List directory",
    "Xem ảnh": "View image",
    "Tìm kiếm web": "Web search",
    "Tải trang": "Fetch page",
    "Tải về": "Download",
    "Tác vụ con": "Subagent",
    "Chờ tác vụ con": "Wait subagent",
    "Nhắn tác vụ con": "Message subagent",
    "Tạo tác vụ con": "Spawn subagent",
    "Đóng tác vụ con": "Close subagent",
    "Ghi chú": "Note",
    "Ghi nhớ": "Memory",
    "Kế hoạch": "Plan",
    "Hoàn tất mục tiêu": "Finish goal",
    "Hỏi người dùng": "Ask user",
    "Lịch hẹn": "Schedule",
    "Hẹn giờ": "Set timer",
    "Hủy hẹn": "Cancel schedule",
    "Tạo công cụ": "Create tool",
    "Chạy song song": "Parallel batch",
    "Chạy tập lệnh": "Run script",
    "Ghi phần mở rộng": "Write plugin",
    "Lệnh lỗi": "Command error",
    "Ngữ cảnh mới": "New context",
    "Đọc tác vụ": "Read task",
    "Nhập vào tác vụ": "Send input",
    "Dừng tác vụ": "Stop task",
    "Dùng mô hình trả phí": "Use paid model",
    "Tìm công cụ": "Search tools",

    // Preview, Processing & Studio Editor
    "Bắt đầu xử lý": "Start processing",
    "Tiếp tục xử lý": "Resume processing",
    "Đang xử lý": "Processing",
    "Đang xử lý ảnh. Vui lòng chờ hoàn tất.": "Processing images. Please wait.",
    "Chưa có trang nào để xử lý.": "No pages to process.",
    "Không có trang nào để xử lý.": "No pages to process.",
    "Chưa có chương để xử lý.": "No chapter loaded to process.",
    "Một chương khác đang được xử lý.": "Another chapter is currently processing.",
    "Chương này đang được xử lý.": "This chapter is currently processing.",
    "Đang lưu vùng giữ nguyên…": "Saving preserved regions…",
    "Bỏ qua lát ảnh": "Skip slice",
    "Đánh dấu vùng giữ nguyên": "Mark preserve region",
    "Đang đánh dấu · Chọn để kết thúc": "Marking · Click to finish",
    "Xóa vùng giữ nguyên": "Clear preserve regions",
    "Xóa vùng giữ nguyên này": "Remove this preserve region",
    "Xóa toàn bộ vùng giữ nguyên của lát ảnh này": "Clear all preserve regions on this slice",
    "Đã bỏ qua": "Skipped",
    "Đã bỏ qua · Chọn để khôi phục": "Skipped · Click to restore",
    "Cần xử lý": "Needs processing",
    "Đã xử lý": "Processed",
    "Trang đang chọn": "Selected page",
    "Trang & lát": "Pages & slices",
    "Điều hướng trang": "Page navigation",
    "Điều hướng trang xử lý": "Processing page navigation",
    "Thuộc tính trang xử lý": "Page processing status",
    "Nhảy đến trang": "Jump to page",
    "Trang trước": "Previous page",
    "Phóng to": "Zoom in",
    "Thu nhỏ": "Zoom out",
    "Thu phóng": "Zoom",
    "Đặt lại thu phóng": "Reset zoom",
    "Vừa khung": "Fit to view",
    "Bấm để vừa khung": "Click to fit view",
    "Đang vừa khung": "Fit to view active",
    "Kích thước thật (100%)": "Actual size (100%)",
    "Xem kích thước thật 1:1": "View 1:1 actual size",
    "Bản đã kết xuất": "Rendered",
    "Đã tẩy nền": "Inpainted",
    "Ảnh gốc": "Original",
    "Ảnh đang xem": "Active view",
    "Đang tải ảnh gốc…": "Loading original image…",
    "Đang tải ảnh đã tẩy nền…": "Loading inpainted image…",
    "Đang tải ảnh có chữ…": "Loading rendered image…",
    "Ngôn ngữ gốc": "Source language",
    "Tiếng Anh": "English",
    "Tiếng Hàn": "Korean",
    "Tiếng Nhật": "Japanese",
    "Tiếng Trung": "Chinese",
    "Chưa rõ": "Unknown",
    "Có chữ": "Has text",
    "theo nguồn truyện": "by comic source",
    "tự nhận": "auto-detected",
    "đã chọn": "selected",
    "đã lưu": "saved",
    "Đang nhận diện…": "Detecting…",
    "Chưa nhận diện được, hãy chọn": "Not detected yet — choose",
    "Chưa nhận diện được ngôn ngữ gốc. Vui lòng chọn tại mục Ngôn ngữ gốc trong trình đơn tùy chọn (⋯).": "Source language not detected yet. Pick it under Source language in the options menu (⋯).",
    "OCR toàn chương": "Chapter OCR",
    "Nhận dạng chữ toàn chương": "Run full-chapter OCR",
    "Chưa chạy OCR toàn chương.": "Full-chapter OCR has not run yet.",
    "Đang nhận dạng": "Recognizing",
    "Đang nhận dạng chữ gốc toàn chương, xem tiến độ tại trình đơn tùy chọn (⋯) ở góc phải.": "Running full-chapter OCR; track progress in the options menu (⋯) at top right.",
    "OCR toàn chương đã hoàn tất.": "Full-chapter OCR complete.",
    "OCR lại": "Re-OCR",
    "Nhận dạng lại bằng OCR": "Re-run OCR on region",
    "Chọn": "Select",
    "Chọn, kéo và thay đổi kích thước vùng chữ.": "Select, move, and resize text boxes.",
    "Vùng chữ nhật": "Rectangle region",
    "Vùng elip": "Ellipse region",
    "Kéo quanh bong bóng thoại để tạo vùng nhận dạng chữ.": "Drag around speech bubble to create a text box.",
    "Kéo quanh bong bóng thoại tròn để tạo vùng nhận dạng chữ.": "Drag around round bubble to create a text box.",
    "Cọ tẩy nền": "Inpaint brush",
    "Tẩy vùng tô": "Eraser brush",
    "Xóa phần vùng tô đã đánh dấu nhầm.": "Erase mistakenly painted mask areas.",
    "Tùy chọn cọ tẩy nền": "Inpaint brush options",
    "Tô vùng cần xóa chữ hoặc vùng cần khôi phục ảnh gốc, sau đó chọn phương thức xử lý.": "Paint areas to clean text or restore original art, then pick a method.",
    "Chọn phương thức xử lý": "Choose processing method",
    "Lựa chọn này áp dụng cho toàn bộ vùng đánh dấu hiện có trên trang.": "This applies to all painted regions on the page.",
    "Xử lý mặc định": "Default clean",
    "Khuyến nghị": "Recommended",
    "Ưu tiên Smart Fill trên nền đồng nhất và chỉ chuyển sang LaMa khi vùng ảnh cần tái tạo phức tạp hơn.": "Prefer Smart Fill on uniform backgrounds and use LaMa only for complex textures.",
    "Vá nền chuyên sâu bằng LaMa": "Deep LaMa inpaint",
    "Bỏ qua Smart Fill và buộc LaMa tái tạo toàn bộ vùng đã đánh dấu. Phương án này có thể mất nhiều thời gian hơn.": "Skip Smart Fill and force LaMa across all painted regions (slower).",
    "Khôi phục ảnh gốc": "Restore original art",
    "Khôi phục nền tự động": "Restore auto-inpaint",
    "Trả vùng đã tô về đúng ảnh gốc, dùng khi quá trình tẩy nền vô tình xóa mất chi tiết tranh. Vùng này được giữ nguyên qua các lần xử lý sau.": "Revert painted area to original art if inpainting erased artwork. Preserved on future runs.",
    "Bàn tay": "Pan",
    "Kéo trang tự do theo mọi hướng.": "Pan around the page freely.",
    "Công cụ chỉnh sửa ảnh và trình bày chữ": "Image cleanup & typesetting tools",
    "Thuộc tính vùng chữ": "Text box inspector",
    "Đóng thuộc tính": "Close inspector",
    "Chọn một vùng chữ trên ảnh để chỉnh sửa.": "Select a text box on the canvas to edit.",
    "Chỉnh sửa chữ gốc, bản dịch và kiểu chữ": "Edit source text, translation, and typography",
    "Nội dung chữ": "Text content",
    "Nội dung gốc / OCR": "Source / OCR text",
    "Bản dịch": "Translation",
    "Chữ gốc": "Source text",
    "Nhập hoặc hiệu chỉnh nội dung gốc…": "Enter or edit source text…",
    "Nhập nội dung bản dịch…": "Enter translation…",
    "Kiểu chữ": "Typography",
    "Phông chữ": "Font",
    "Chọn phông chữ": "Choose font",
    "Tìm phông chữ": "Search fonts",
    "Tìm phông chữ…": "Search fonts…",
    "Danh sách phông chữ": "Font list",
    "Không tìm thấy phông phù hợp.": "No matching fonts found.",
    "Cỡ chữ": "Font size",
    "Tự động vừa vùng": "Auto-fit box",
    "Căn ngang": "Horizontal align",
    "Căn dọc": "Vertical align",
    "Trái": "Left",
    "Giữa": "Center",
    "Phải": "Right",
    "Trên": "Top",
    "Dưới": "Bottom",
    "Màu chữ": "Text color",
    "Màu viền chữ": "Stroke color",
    "Viền chữ": "Outline",
    "Độ dày viền chữ": "Outline width",
    "Tự động tương phản": "Auto contrast",
    "Đen": "Black",
    "Trắng": "White",
    "Đỏ": "Red",
    "Vàng": "Yellow",
    "Chọn màu tùy chỉnh": "Pick custom color",
    "In đậm chữ": "Bold text",
    "Hiệu ứng": "Style role",
    "Lời thoại": "Dialogue",
    "Suy nghĩ": "Thought",
    "Dẫn truyện": "Narration",
    "Chiêu thức": "Special move",
    "Kinh dị": "Horror",
    "Lãng mạn": "Romance",
    "Khác": "Other",
    "Mặc định (Comic)": "Default (Comic)",
    "Mặc định (Truyện tranh)": "Default (Comic)",
    "Kiểu chữ mẫu": "Style presets",
    "Chưa có mẫu": "No presets yet",
    "Chọn kiểu mẫu…": "Choose preset…",
    "Lưu kiểu chữ hiện tại thành mẫu": "Save current style as preset",
    "Lưu kiểu hiện tại thành mẫu": "Save current style as preset",
    "Lưu kiểu chữ của vùng đang chọn thành mẫu": "Save selected box style as preset",
    "Tên mẫu (vd: Lời thoại)…": "Preset name (e.g. Dialogue)…",
    "Tên kiểu chữ mẫu mới": "New style preset name",
    "Đặt tên cho kiểu chữ mẫu": "Name the style preset",
    "Áp dụng": "Apply",
    "Lưu": "Save",
    "Xác nhận": "Confirm",
    "Xóa vùng chữ": "Delete text box",
    "Xóa vùng chữ đang chọn": "Delete selected text box",
    "Xóa vùng đang chọn": "Delete selected region",
    "Xóa kiểu chữ mẫu đang chọn": "Delete selected preset",
    "Đã lưu": "Saved",
    "Chưa lưu": "Unsaved",
    "Có thay đổi chưa lưu": "Unsaved changes",
    "Đang lưu…": "Saving…",
    "Đang lưu thay đổi": "Saving changes",
    "Tất cả thay đổi đã được lưu": "All changes saved",
    "Lưu thất bại": "Save failed",
    "Lưu thay đổi thất bại": "Failed to save changes",
    "Trạng thái trang": "Page status",
    "Soát chữ": "Proofing",
    "Soát chữ toàn chương": "Chapter proofing",
    "Soát chữ toàn chương (Ctrl+F)": "Chapter proofing (Ctrl+F)",
    "Soát chữ: tìm / thay thế": "Proofing: find / replace",
    "Đóng bảng soát chữ": "Close proofing panel",
    "Tìm (Ctrl+F)…": "Find (Ctrl+F)…",
    "Thay bằng (Ctrl+H)…": "Replace with (Ctrl+H)…",
    "Thay bằng": "Replace with",
    "Thay thế": "Replace",
    "Thay tất cả": "Replace all",
    "Phân biệt hoa thường": "Case sensitive",
    "Tìm trong": "Search in",
    "Chưa dịch": "Untranslated",
    "Chưa có vùng chữ nào.": "No text boxes yet.",
    "Không có vùng nào khớp.": "No matching regions.",
    "Không có chỗ nào để thay.": "No matches to replace.",
    "Nhập chữ cần tìm trước.": "Enter search text first.",
    "Chọn mọi vùng đang hiện": "Select all visible regions",
    "Kiểu mẫu…": "Preset…",
    "Áp": "Apply",
    "Phím tắt": "Shortcuts",
    "Phím tắt (?)": "Shortcuts (?)",
    "Mở / đóng bảng phím tắt": "Toggle shortcuts sheet",
    "Hoàn tác (Ctrl+Z)": "Undo (Ctrl+Z)",
    "Làm lại (Ctrl+Y)": "Redo (Ctrl+Y)",
    "Hoàn tác / làm lại": "Undo / Redo",
    "Xuất chương": "Export chapter",
    "Tùy chọn xuất": "Export options",
    "Định dạng": "Format",
    "Bề rộng": "Width",
    "Chất lượng": "Quality",
    "Giữ nguyên": "Original",
    "Không cắt": "No split",
    "Cắt trang cao tối đa": "Max split height",
    "Tiền tố tên tệp": "Filename prefix",
    "Gói": "Archive",
    "Đang kết xuất chương…": "Rendering chapter…",
    "LaMa đang xử lý…": "LaMa is processing…",
    "Thêm thao tác": "More actions",
    "Biên tập bản dịch": "Edit translations",
    "Kiểm tra chất lượng": "Quality check",
    "Xử lý ảnh": "Image cleanup",
    "Hoàn tất": "Done",
    "Hoàn tất!": "Done!",
    "Xong": "Done",
    "Cứu:": "Rescued:",
    "Mạng bật.": "Network on.",
    "Mạng tắt.": "Network off.",
    "Xóa tài khoản Manga Cloud? Địa chỉ email và phiên đăng nhập sẽ bị xóa. Số dư còn lại vẫn được bảo lưu và khôi phục khi bạn đăng nhập lại bằng đúng email này.": "Delete your Manga Cloud account? Your email address and sign-in sessions will be removed. Any remaining balance is preserved and restored if you sign back in with the same email.",
    "Đã khôi phục ảnh gốc ở các vùng được đánh dấu.": "Restored original art on marked regions.",
    "Đã làm sạch các vùng được đánh dấu.": "Cleaned marked regions.",
    "Không có vùng ảnh nào được cập nhật.": "No image regions were updated.",
    "Không còn thao tác để hoàn tác.": "Nothing to undo.",
    "Không còn thao tác để làm lại.": "Nothing to redo.",
  });

  const PATTERNS = [
    // Numbers / counters
    [/^(\d+) bước$/, "$1 steps"],
    [/^(\d+) lỗi$/, "$1 failed"],
    [/^(\d+) lượt$/, "$1 turns"],
    [/^(\d+) công cụ$/, "$1 tools"],
    [/^(\d+) mục$/, "$1 items"],
    [/^ · (\d+) biến$/, " · $1 vars"],
    [/^(\d+) lần cứu$/, "$1 rescues"],
    [/^Đang bọc tự động \((\d+) model từng gặp lỗi\)$/, "Auto-wrapped ($1 models rescued from errors)"],
    [/^Quy tắc bọc:\s*(.+)$/, (_, r) => `Wrapped rules: ${translateRules(r)}`],
    [/^Cứu:\s*(.+)$/, (_, r) => `Rescued: ${r}`],
    [/^Lỗi:\s*(.+)$/, (_, r) => `Error: ${r}`],
    [/^Thêm mô hình(?:\s+(\d+))?$/, (_, n) => (n ? `More models (${n})` : "More models")],
    [/^Tìm nhanh trong (\d+) mô hình… \(Enter dùng tên nhập\)$/, "Search $1 models… (Enter to use typed ID)"],
    [/^Đang tải (\d+)\/(\d+) mô hình…$/, "Loading $1/$2 models…"],
    [/^Chạy hàng đợi \((\d+) chương\)$/, "Run queue ($1 chapters)"],
    [/^Danh sách liên kết chương \((.*)$/, "Chapter link list ($1"],
    [/^(\d+)\/(\d+) hoàn tất$/, "$1/$2 completed"],
    [/^Đang xử lý chương #(\d+):\s*(.+)$/, "Processing chapter #$1: $2"],
    [/^chương (\d+\/\d+)$/, "chapter $1"],
    [/^Đang chạy (.+)$/, "Running $1"],
    [/^Dừng do lỗi:\s*(.+)$/, "Stopped on error: $1"],
    [/^Đã quét (\d+) chương · đang chọn (\d+)[–-](\d+)$/, "Scanned $1 chapters · selected $2–$3"],
    [/^\s*\(đang nạp (\d+) chương đầu; dùng thanh Từ\/Đến bên dưới để chọn các chương tiếp theo\)$/, " (loaded first $1 chapters; use From/To below for more)"],
    [/^Tìm thấy (\d+) chương đúng của bộ truyện(.*)$/, "Found $1 chapters for this series$2"],
    [/^Hàng đợi hoàn tất \((\d+)\/(\d+) chương\)$/, "Queue completed ($1/$2 chapters)"],
    [/^Hàng đợi xong (\d+)\/(\d+) chương \((\d+) lỗi\)\.$/, "Queue finished $1/$2 chapters ($3 failed)."],
    [/^Hoàn tất (\d+)\/(\d+) chương · (\d+) lỗi$/, "Finished $1/$2 chapters · $3 errors"],
    [/^Hàng đợi dừng do lỗi \((\d+)\/(\d+)\)$/, "Queue stopped on error ($1/$2)"],
    [/^Đã hủy hàng đợi \((\d+)\/(\d+) chương đã xong\)$/, "Queue canceled ($1/$2 chapters completed)"],
    [/^Mỗi hàng đợi tối đa (\d+) chương \(hiện đang có (\d+)\)\.$/, "Maximum $1 chapters per queue (currently $2)."],
    [/^Đã nạp lại (\d+) chương chưa hoàn tất\.$/, "Reloaded $1 incomplete chapters."],
    [/^·\s*chi phí\s*~\$(.+)$/, "· cost ~$$$1"],
    [/^Tiến độ kế hoạch · (\d+)\/(\d+) hoàn tất$/, "Plan progress · $1/$2 done"],
    [/^(.+?) · chọn mô hình$/, "$1 · select model"],
    [/^Suy luận:\s*(.+)$/, (_, v) => `Reasoning: ${translateText(v)}`],
    [/^Mục tiêu:\s*(.+)$/, "Goal: $1"],
    [/^Kỹ năng:\s*(.+)$/, "Skills: $1"],
    [/^Đã tắt:\s*(.+)$/, "Disabled: $1"],
    [/^(.+?) thay bằng (.+)$/, "$1 replaced by $2"],
    [/^Công cụ đã tạo:\s*(.+)$/, "Created tools: $1"],
    [/^Công cụ dự án chưa cho phép:\s*(.+)$/, "Untrusted workspace tools: $1"],
    [/^Phần mở rộng (.+?) \((.+?)\) · (.+)$/, (_, n, sc, st) => `Plugin ${n} (${translateText(sc)}) · ${translateText(st)}`],
    [/^Hook: (\d+) của bạn, (\d+) của dự án(.*)$/, (_, u, w, tail) => `Hooks: ${u} user, ${w} project${tail ? " (untrusted)" : ""}`],
    [/^MCP ([^\s]+) · (.+)$/, (_, name, st) => `MCP ${name} · ${translateText(st)}`],
    [/^Tác vụ con (.*?) hoàn tất:\s*(.+?) · (\d+) lệnh gọi$/, "Subagent $1 done: $2 · $3 tool calls"],
    [/^Tác vụ con:\s*(.+?)\. Lệnh: (.+)$/, "Subagents: $1. Commands: $2"],
    [/^Khóa API (\d+\/\d+) \((.+?)\) bị giới hạn$/, "API key $1 ($2) rate-limited"],
    [/^Chuyển sang khóa API (\d+\/\d+) và tiếp tục\?$/, "Switch to API key $1 and continue?"],
    [/^Đổi sang khóa API (\d+)$/, "Switch to API key $1"],
    [/^Luôn cho phép (.+)$/, "Always allow $1"],
    [/^Đã lưu luật:\s*(.+)$/, "Saved rule: $1"],
    [/^Việc cần làm trong (.+?)… \(\/ xem lệnh\)$/, "Task in $1… (/ for commands)"],
    [/^Tác vụ con\s+(.+?)\s+yêu cầu$/, "Subagent $1 requests"],
    [/^(.+?)\s+Mạng\s+(bật|tắt)\.$/, (_, head, st) => `${translateText(head)} Network ${st === "bật" ? "on" : "off"}.`],
    [/^Cho phép (.+?) (.+)\?$/, "Allow $1 $2?"],
    [/^Tác vụ con (.+?) yêu cầu (.+?) (.+)\?$/, "Subagent $1 requests $2 $3?"],
    [/^Bạn trả lời:\s*(.+)$/, "You answered: $1"],
    [/^Đính kèm:\s*(.+)$/, "Attached: $1"],
    [/^Theo cài đặt \((.+)\)$/, "From settings ($1)"],
    [/^Dùng:\s*(.+)$/, "Use: $1"],
    [/^Đã kết nối · (\d+) khóa$/, "Connected · $1 keys"],
    [/^(\d+) nhà cung cấp sẵn sàng$/, "$1 providers ready"],
    [/^Đã tải (\d+) mô hình\.$/, "Loaded $1 models."],
    [/^Đã lưu nhà cung cấp (.+)$/, "Saved provider $1"],
    [/^Đã xóa nhà cung cấp (.+)$/, "Removed provider $1"],
    [/^(\d+) vùng chữ · (\d+(?:\/\d+)?) đã dịch$/, "$1 boxes · $2 translated"],
    [/^(\d+) vùng · (\d+) đã dịch$/, "$1 boxes · $2 translated"],
    [/^(\d+) \/ (\d+) vùng$/, "$1 / $2 regions"],
    [/^Trang (\d+) · Lát (\d+)\/(\d+)$/, "Page $1 · Slice $2/$3"],
    [/^Lát (\d+)$/, "Slice $1"],
    [/^(\d+) vùng giữ nguyên$/, "$1 preserve regions"],
    [/^(\d+)\/(\d+) lát được chọn$/, "$1/$2 slices selected"],
    [/^Đang xử lý (\d+)\/(\d+)…$/, "Processing $1/$2…"],
    [/^Đã xử lý (\d+)\/(\d+)…$/, "Processed $1/$2…"],
    [/^Đang kết xuất (\d+)\/(\d+)…$/, "Rendering $1/$2…"],
    [/^Đã kết xuất (\d+) trang\.$/, "Rendered $1 pages."],
    [/^Đang tải (\d+) tệp…$/, "Uploading $1 files…"],
    [/^Áp \((\d+)\)$/, "Apply ($1)"],
    [/^Chương (.+)$/, "Chapter $1"],
    [/^Mở lại chương (.+)$/, "Reopen chapter $1"],
    [/^Ngôn ngữ gốc:\s*(.+)$/, (_, l) => `Source language: ${translateText(l)}`],
    [/^Xóa kỹ năng (.+)\?$/, "Delete skill $1?"],
    [/^Xóa máy chủ MCP (.+)\?$/, "Delete MCP server $1?"],
    [/^Đã lưu kỹ năng (.+)\.$/, "Saved skill $1."],
    [/^Đã xóa (.+)\.$/, "Deleted $1."],
    [/^Đã cài đặt (.+?)\. Vui lòng đọc SKILL\.md trước khi sử dụng\.$/, "Installed $1. Read SKILL.md before using."],
    [/^Lưu ở (.+)$/, "Saved at $1"],
    [/^Không thể tải danh sách mô hình:\s*(.*)$/, "Could not fetch model list: $1"],
    [/^Không thể lưu nhà cung cấp:\s*(.*)$/, "Could not save provider: $1"],
    [/^Không thể xóa nhà cung cấp:\s*(.*)$/, "Could not remove provider: $1"],
    [/^Không tải được chương:\s*(.*)$/, "Failed to load chapter: $1"],
    [/^Không tải được danh sách chương:\s*(.*)$/, "Failed to load chapters: $1"],
    [/^Không thể chạy Chế độ AI:\s*(.*)$/, "Could not start AI Mode: $1"],
    [/^Không thể chạy hàng đợi AI:\s*(.*)$/, "Could not start AI queue: $1"],
    [/^Không quét được mục lục:\s*(.*)$/, "Failed to scan series TOC: $1"],
    [/^Không hủy được:\s*(.*)$/, "Failed to cancel: $1"],
    [/^Không hủy được hàng đợi:\s*(.*)$/, "Failed to cancel queue: $1"],
    [/^Không lưu được:\s*(.*)$/, "Failed to save: $1"],
    [/^Máy chủ trả về lỗi HTTP (.+)$/, "Server returned HTTP $1"],
    [/^([\d.,]+[kKmMbB]?) vào · ([\d.,]+[kKmMbB]?) ra(.*)$/, "$1 in · $2 out$3"],
    [/^Bỏ mọi vùng đã làm sạch bằng cọ trên lát (\d+)\? Lát sẽ trở về kết quả xử lý tự động\.$/, "Discard all brush-cleaned regions on slice $1? It will revert to auto processing."],
    [/^([\s\S]+?)\nVẫn xuất chương\? Những vùng đó sẽ để trống chữ\.$/, "$1\nExport chapter anyway? Those boxes will have blank text."],
    [/^Đã áp kiểu "(.+)" cho (\d+) vùng\. Ctrl\+Z để hoàn tác\.$/, 'Applied style "$1" to $2 boxes. Ctrl+Z to undo.'],
    [/^Đã áp kiểu "(.+)"\.$/, 'Applied style "$1".'],
    [/^Đã lưu kiểu "(.+)"\.$/, 'Saved style "$1".'],
    [/^Đã xóa kiểu "(.+)"\.$/, 'Deleted style "$1".'],
    [/^Đã thay (\d+) chỗ trong (\d+) vùng\. Ctrl\+Z để hoàn tác\.$/, "Replaced $1 occurrences across $2 boxes. Ctrl+Z to undo."],
    [/^Chưa có kiểu chữ mẫu số (\d+)\. Lưu kiểu mẫu trong mục Kiểu chữ\.$/, "No preset #$1 yet. Save one under Typography."],
    [/^Lỗi khi (.+?) \((\d+)\):\s*(.+)$/, "Error during $1 ($2): $3"],
    [/^(.+?)\s+·\s+(chỉ đọc|ghi thư mục|không cô lập)(\s+·\s+mạng)?$/, (_, m, sb, net) => `${translateText(m)} · ${translateText(sb)}${net ? " · network" : ""}`],
    [/^(?:chạy|đọc|sửa|tạo|chờ|nhắn|đóng|ghi nhớ|xem|tìm|tìm web|tải|mở|gọi|nhập vào|dừng|hẹn)\s+\d+\s+.+$/i, (whole) => translateTallyPhrase(whole)],
  ];

  const TALLY_VERBS = {
    "chạy": "run", "đọc": "read", "sửa": "edit", "tạo": "spawn", "chờ": "wait",
    "nhắn": "message", "đóng": "close", "ghi nhớ": "save", "xem": "list", "tìm": "search",
    "tìm web": "web-search", "tải": "fetch", "mở": "open", "gọi": "call", "nhập vào": "input",
    "dừng": "stop", "hẹn": "schedule",
  };
  const TALLY_NOUNS = {
    "lệnh": "commands", "tệp": "files", "công cụ": "tools", "tác vụ con": "subagents",
    "mục": "items", "thư mục": "folders", "lần": "times", "trang": "pages", "kỹ năng": "skills",
    "công cụ mcp": "MCP tools", "tập lệnh": "scripts", "lô song song": "parallel batches",
    "tác vụ": "tasks", "lượt": "turns",
  };

  function translateTallyPhrase(whole) {
    const parts = String(whole).split(",").map((part) => {
      const m = /^\s*(chạy|đọc|sửa|tạo|chờ|nhắn|đóng|ghi nhớ|xem|tìm web|tìm|tải|mở|gọi|nhập vào|dừng|hẹn)\s+(\d+)\s+(.+?)\s*$/i.exec(part);
      if (!m) return part.trim();
      const v = TALLY_VERBS[m[1].toLowerCase()] || m[1];
      const n = TALLY_NOUNS[m[3].toLowerCase()] || m[3];
      return `${v} ${m[2]} ${n}`;
    });
    const joined = parts.join(", ");
    return joined.charAt(0).toUpperCase() + joined.slice(1);
  }

  function translateRules(text) {
    return String(text || "")
      .replace(/(^|[^A-Za-zÀ-ỹ0-9_])Ảnh(?=$|[^A-Za-zÀ-ỹ0-9_])/g, "$1Images")
      .replace(/(^|[^A-Za-zÀ-ỹ0-9_])Gộp system(?=$|[^A-Za-zÀ-ỹ0-9_])/g, "$1Fold system")
      .replace(/(^|[^A-Za-zÀ-ỹ0-9_])Bỏ:/g, "$1Strip:");
  }

  function translateText(raw) {
    const str = String(raw ?? "");
    const m = /^(\s*)([\s\S]*?)(\s*)$/.exec(str);
    if (!m) return str;
    const [, lead, core, trail] = m;
    if (!core) return str;
    if (EXACT[core]) return `${lead}${EXACT[core]}${trail}`;
    if (!viRegex.test(core) && !/^(?:MCP\s|Hook:|·\s*\d)/i.test(core)) return str;
    for (let i = 0; i < PATTERNS.length; i += 1) {
      const [rx, repl] = PATTERNS[i];
      if (rx.test(core)) {
        return `${lead}${core.replace(rx, repl)}${trail}`;
      }
    }
    return str;
  }

  let currentLang = DEFAULT_LANG;
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved && SUPPORTED.has(saved)) currentLang = saved;
  } catch (_) {}

  const originText = new WeakMap();
  const originAttrs = new WeakMap();
  let observer = null;
  let applyDepth = 0;

  function beginApply() {
    applyDepth += 1;
  }

  function endApply() {
    if (applyDepth > 0) applyDepth -= 1;
    if (applyDepth === 0 && observer) observer.takeRecords();
  }

  const TEXT_SKIP_SELECTOR = "[data-no-i18n], .agent-user, .agent-prose, .agent-pre, .agent-think-body, .agent-target, .ocr-textarea, .review-translation-input, .review-inline-translation, .proofing-source, .proofing-translation, .review-proof-source, .review-proof-translation, .translation-textarea";

  function isHardSkippedElement(el) {
    if (!el || el.nodeType !== 1) return false;
    if (el.id === "lang-toggle") return true;
    const tag = (el.tagName || "").toUpperCase();
    return tag === "SCRIPT" || tag === "STYLE" || tag === "NOSCRIPT" || tag === "SVG";
  }

  function isSkippedTextElement(el) {
    if (isHardSkippedElement(el)) return true;
    if (el.tagName === "TEXTAREA") return true;
    if (el.closest?.(TEXT_SKIP_SELECTOR)) return true;
    if (el.tagName === "CODE" && !EXACT[el.textContent?.trim()]) return true;
    return false;
  }

  function isSkippedAttrElement(el) {
    if (isHardSkippedElement(el)) return true;
    if (el.closest?.("[data-no-i18n]")) return true;
    return false;
  }

  function translateForNode(parent, source) {
    if (parent?.id === "agent-skill-install" && source.trim() === "Cài đặt") {
      return "Install";
    }
    return translateText(source);
  }

  function processTextNode(node) {
    if (!node || node.nodeType !== 3) return;
    const parent = node.parentElement;
    if (!parent || isSkippedTextElement(parent)) return;
    const val = node.nodeValue || "";
    if (!val.trim()) {
      if (currentLang === "en") originText.delete(node);
      return;
    }

    if (currentLang === "en") {
      const entry = originText.get(node);
      if (!entry || val !== entry.translated) {
        const next = translateForNode(parent, val);
        originText.set(node, { orig: val, translated: next });
        if (next !== val) node.nodeValue = next;
      }
    } else if (originText.has(node)) {
      const entry = originText.get(node);
      if (entry && val === entry.translated && entry.orig !== val) {
        node.nodeValue = entry.orig;
      }
      originText.delete(node);
    }
  }

  function processAttributes(el) {
    if (!el || el.nodeType !== 1 || isSkippedAttrElement(el)) return;
    let map = originAttrs.get(el);
    ATTRS.forEach((attr) => {
      if (!el.hasAttribute(attr)) {
        if (map && attr in map) delete map[attr];
        return;
      }
      const val = el.getAttribute(attr) || "";
      if (!val.trim()) {
        if (map && attr in map && currentLang === "en") delete map[attr];
        return;
      }
      if (currentLang === "en") {
        if (!map) {
          map = Object.create(null);
          originAttrs.set(el, map);
        }
        const entry = map[attr];
        if (!entry || val !== entry.translated) {
          const next = translateText(val);
          map[attr] = { orig: val, translated: next };
          if (next !== val) el.setAttribute(attr, next);
        }
      } else if (map && map[attr]) {
        const entry = map[attr];
        if (val === entry.translated && entry.orig !== val) {
          el.setAttribute(attr, entry.orig);
        }
        delete map[attr];
      }
    });
  }

  function applyTree(root) {
    if (!root) return;
    beginApply();
    try {
      if (root.nodeType === 3) {
        processTextNode(root);
        return;
      }
      if (root.nodeType !== 1 && root !== document.body) return;
      if (root.nodeType === 1 && isHardSkippedElement(root)) return;
      if (root.nodeType === 1) processAttributes(root);
      const walker = document.createTreeWalker(
        root,
        NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT,
        {
          acceptNode(n) {
            if (n.nodeType === 1 && (isHardSkippedElement(n) || n.hasAttribute?.("data-no-i18n"))) {
              return NodeFilter.FILTER_REJECT;
            }
            return NodeFilter.FILTER_ACCEPT;
          },
        },
      );
      let node = walker.nextNode();
      while (node) {
        if (node.nodeType === 1) processAttributes(node);
        else if (node.nodeType === 3) processTextNode(node);
        node = walker.nextNode();
      }
    } finally {
      endApply();
    }
  }

  function syncLangControls() {
    document.documentElement.lang = currentLang;
    document.documentElement.dataset.uiLang = currentLang;
    const toggle = document.getElementById("lang-toggle");
    if (toggle) {
      toggle.textContent = currentLang.toUpperCase();
      toggle.setAttribute(
        "aria-label",
        currentLang === "en" ? "Switch UI language to Vietnamese" : "Chuyển ngôn ngữ giao diện sang tiếng Anh",
      );
      toggle.setAttribute(
        "title",
        currentLang === "en" ? "UI Language: English (click for VI)" : "Ngôn ngữ giao diện: Tiếng Việt (bấm đổi EN)",
      );
    }
    document.querySelectorAll("[data-ui-lang-select]").forEach((sel) => {
      if (sel.value !== currentLang) sel.value = currentLang;
    });
  }

  function stopObserver() {
    if (!observer) return;
    observer.disconnect();
    observer = null;
  }

  function setUiLang(lang) {
    const normalized = SUPPORTED.has(lang) ? lang : DEFAULT_LANG;
    currentLang = normalized;
    try {
      localStorage.setItem(STORAGE_KEY, normalized);
    } catch (_) {}
    if (normalized !== "en") stopObserver();
    if (document.body) applyTree(document.body);
    if (normalized === "en") startObserver();
    syncLangControls();
    window.dispatchEvent(new CustomEvent("app-lang-change", { detail: { lang: normalized } }));
    return normalized;
  }

  function startObserver() {
    if (currentLang !== "en" || observer || !document.body) return;
    observer = new MutationObserver((mutations) => {
      if (applyDepth > 0 || currentLang !== "en") return;
      beginApply();
      try {
        for (const m of mutations) {
          if (m.type === "characterData") {
            processTextNode(m.target);
          } else if (m.type === "attributes") {
            processAttributes(m.target);
          } else if (m.type === "childList") {
            m.addedNodes.forEach((n) => {
              if (n.nodeType === 3) processTextNode(n);
              else if (n.nodeType === 1) applyTree(n);
            });
          }
        }
      } finally {
        endApply();
      }
    });
    observer.observe(document.body, {
      childList: true,
      subtree: true,
      characterData: true,
      attributes: true,
      attributeFilter: ATTRS,
    });
  }

  window.t = (text) => (currentLang === "en" ? translateText(text) : String(text ?? ""));
  window.getUiLang = () => currentLang;
  window.setUiLang = setUiLang;
  window.i18n = { getLang: () => currentLang, setLang: setUiLang, t: window.t };

  const nativeConfirm = window.confirm;
  if (typeof nativeConfirm === "function") {
    window.confirm = (message) => nativeConfirm.call(window, window.t(message));
  }

  document.documentElement.lang = currentLang;
  document.documentElement.dataset.uiLang = currentLang;

  document.addEventListener("DOMContentLoaded", () => {
    syncLangControls();
    document.getElementById("lang-toggle")?.addEventListener("click", () => {
      setUiLang(currentLang === "en" ? "vi" : "en");
    });
    if (currentLang === "en" && document.body) {
      applyTree(document.body);
    }
    startObserver();
  });
})();
