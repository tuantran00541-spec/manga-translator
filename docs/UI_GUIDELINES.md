# UI System Guidelines

## Product stages

The application exposes one primary workspace at a time. Use these Vietnamese stage names consistently:

1. **Nhập nội dung** — URL, local files, recent chapters.
2. **Xử lý ảnh** — page/slice selection, exclusions, automatic processing.
3. **Kiểm tra chất lượng** — manual correction and AI-assisted QC.
4. **Biên tập bản dịch** — text regions, OCR content, translation, styling, rendering.

Landing controls must not remain visible behind Preview, Review, or Editor. Stage switching is owned by `ui-shell.js`.

## Product language

Use neutral, professional Vietnamese. Internal names may remain English in code/API, but user-facing copy should not expose implementation vocabulary unless it is an established technical term such as OCR, API, Gemini, PNG, ZIP, or CBZ.

Preferred terms:

- chapter → **chương**
- text object → **vùng chữ**
- clean image → **ảnh đã xử lý**
- AI QC → **kiểm tra bằng AI** / **kiểm tra chất lượng bằng AI**
- repaint → **xử lý vùng đánh dấu**
- render → **kết xuất**
- excluded region → **vùng loại trừ**
- manual mask → **vùng chỉnh sửa thủ công**

Buttons should use **verb + object**. Status text should be short and factual. Avoid conversational copy such as “Ổn rồi…”, “dịch dở”, or “AI rà…”.

## Hierarchy

Each workspace should contain, in order:

1. global header + stage indicator;
2. one workspace command surface;
3. compact page navigation;
4. the primary image/canvas;
5. contextual side panel only when the task requires it.

Do not create a second independent toolbar for a feature. Add feature configuration to the global Settings drawer and keep only the feature action/status inside the active workspace.

## Actions

- **Primary**: advances or commits the main task (process pages, process marked region, render translation).
- **Secondary**: tools and reversible actions.
- **Danger**: deletion/reset actions. Keep visually separate from the primary action.

A long-running operation must disable every control that can invalidate the operation's snapshot. For AI QC this includes brush mode, clearing marks, brush size, repaint/reset, page navigation, and advancing to Editor.

## Responsive behavior

- Desktop: command bars may use a single horizontal row when space permits.
- Tablet: controls may wrap into a second row inside the same command surface.
- Mobile (`<= 560px`): page position occupies the first navigation row; Previous/Next occupy the second row. Never rely on horizontal page scrolling for core navigation.
- No workspace may create document-level horizontal overflow.

## Settings

Configuration that is not part of the current editing gesture belongs in the global **Cài đặt** drawer. Gemini API key management and privacy information live there. The Review workspace only shows AI readiness and the **Kiểm tra bằng AI** action.

## Source ownership

- `app/static/css/ui-system.css`: global layout, tokens, cross-stage component styling, responsive rules.
- `app/static/js/ui-shell.js`: stage visibility, global context, settings drawer.
- Stage files (`preview.js`, `review*.js`, `editor.js`): task behavior and stage-specific DOM only.

Do not add another wrapper layer around a stage without first checking whether the shared shell can own the behavior.
