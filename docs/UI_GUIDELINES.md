# UI System Guidelines — v0.3 Workstation

These rules define the target UI architecture for `feat/v03-workstation-ui`. See `docs/UI_V03_WORKSTATION_AUDIT.md` for the migration map and removal list.

## Product model

The application is a creative workstation, not a dashboard. Once a chapter is open, the manga canvas is the primary content and application chrome must remain quieter than the canvas.

Stages remain:

1. **Nhập nội dung** — URL, local files, recent chapters.
2. **Xử lý ảnh** — page/slice selection, exclusions, automatic processing.
3. **Kiểm tra chất lượng** — manual correction and AI-assisted QC.
4. **Biên tập bản dịch** — text regions, OCR, translation, styling, rendering/export.

Stage semantics are stable; their old horizontal presentation is not.

## Workbench layout

Desktop workspaces use one shared structure:

1. **App Rail** — stage navigation and global entry points.
2. **Page Navigator** — shared page/slice navigation and state.
3. **Canvas** — dominant primary workspace.
4. **Context Inspector** — properties/actions for the current selection.
5. Optional thin **Top Command Bar** — chapter identity, job/save state, zoom, and the primary action.

Preview, Review, and Editor must not create their own independent page-navigation systems.

Panels should be collapsible where practical. On narrow screens, side panels become drawers/sheets instead of squeezing the canvas into unusable widths.

## Context follows selection

The Context Inspector changes according to what the user selects:

- no selection → page/chapter state and main actions;
- page → page processing/review state;
- detected/review region → provenance, QC state, repaint/review actions;
- text object → OCR, translation, geometry, typography, appearance and object actions.

Do not show every available control at once.

## Visual hierarchy

Use at most four persistent surface levels:

1. app background;
2. panel/sidebar;
3. raised interactive surface;
4. popover/modal/overlay.

Avoid card nesting as a layout technique. Cards are for bounded data objects, not every container.

Navigation should be visually dimmer than the canvas/content. Semantic state may be stronger when attention is required.

## Tokens

New UI uses one token source only.

Required scales:

- spacing: `4 / 8 / 12 / 16 / 24 / 32`;
- radii: small / medium / large;
- control heights: compact / standard;
- surfaces: app / panel / raised / overlay;
- text: primary / secondary / muted;
- semantic: accent / success / warning / danger / review.

Compatibility aliases such as `--ink`, `--panel`, `--paper`, `--blue` may exist only during migration and must be deleted when their last consumer is migrated.

New component CSS must not use `!important` except for a documented browser/third-party normalization case.

## Actions

- **Primary** — advances or commits the main task in the current context.
- **Secondary** — reversible tools/actions.
- **Danger** — deletion/reset; visually separated from the primary action.
- **Gesture tools** — select, brush, draw region, pan/zoom; these belong close to the canvas.

Persistent configuration is not a toolbar action. Provider keys, privacy, defaults and non-gesture settings belong in Settings or the Inspector.

Buttons use concise **verb + object** labels. Icon-only controls require an accessible label and tooltip.

## State

Critical state must never be represented by color alone. Pair semantic color with text and/or icon for:

- verified;
- needs review;
- OCR failure/stale OCR;
- processing;
- rendered;
- skipped;
- error.

Long-running operations must disable controls that could invalidate the operation snapshot while preserving safe navigation only when the backend semantics allow it.

## Canvas

The canvas owns:

- page image;
- selection overlays;
- direct manipulation;
- brush/draw overlays;
- QC highlights;
- zoom/pan affordances.

Do not wrap the canvas in decorative card chrome unless the boundary communicates a real state or interaction.

## Navigation

Page index, Prev/Next, jump, selection continuity and stage-to-stage canonical page mapping are shared behavior.

There must be exactly one implementation of page navigation logic. Stage modules consume it; they do not recreate it.

## Settings

Global configuration lives in the application Settings surface. Provider/API settings use consistent sections instead of provider-specific ad-hoc UI layouts.

Privacy text must clearly state when source/clean images or contact sheets are sent to an external provider.

## Responsive behavior

- Desktop: rail + page navigator + canvas + inspector.
- Medium widths: one sidebar may collapse; canvas remains primary.
- Tablet/mobile: sidebars become drawers/sheets and only task-critical controls stay visible.
- No document-level horizontal overflow.
- Core navigation cannot depend on a horizontally scrolling thumbnail strip.

## Accessibility and interaction

- Visible keyboard focus everywhere.
- Minimum practical target size follows WCAG guidance; visually compact controls still need usable hit areas.
- Repetitive editing tools should support keyboard shortcuts where safe.
- Hover cannot be the only way to reveal destructive or required actions.
- Motion is short and functional; respect reduced-motion preferences.

## Source ownership target

During migration, legacy files may coexist. Final v0.3 ownership should converge toward:

- `tokens.css` — tokens only;
- `workbench.css` — shell, rail, sidebars, top bar, responsive layout;
- `components.css` — controls, fields, badges, progress, settings, toast, tooltip;
- `canvas.css` — canvas and overlay primitives;
- small stage CSS modules — only stage-specific visual behavior;
- shared JS workbench/page-navigation components;
- stage JS modules — domain behavior, not duplicate shell construction.

A migrated component is not complete until the conflicting legacy CSS/DOM it replaces is removed.

## Product language

Use neutral, professional Vietnamese. Internal names may remain English in code/API; user-facing copy should avoid unnecessary implementation vocabulary.

Preferred terms:

- chapter → **chương**
- text object → **vùng chữ**
- clean image → **ảnh đã xử lý**
- AI QC → **kiểm tra bằng AI** / **kiểm tra chất lượng bằng AI**
- repaint → **xử lý vùng đánh dấu**
- render → **kết xuất**
- excluded region → **vùng loại trừ**
- manual mask → **vùng chỉnh sửa thủ công**
