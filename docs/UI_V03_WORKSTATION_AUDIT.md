# v0.3 Workstation UI Audit

Status: design/architecture baseline for `feat/v03-workstation-ui`.

## Goal

Replace the current layered UI skin with one coherent creative-workstation interface. The manga page/canvas is the primary content. Navigation, controls, AI/QC status, and editing properties support the canvas instead of competing with it.

Reference patterns:
- Linear: calmer, consistent headers/navigation; dim navigation so primary content wins.
- Figma: navigation/pages on the left, canvas in the center, contextual properties on the right, compact tool surface.
- VS Code: workbench shell with stable activity/navigation regions and a dominant editor region.

This is not a visual reskin. Existing UI that conflicts with the new information architecture may be removed or replaced.

## Current baseline

Static audit of `main` after v0.2:

- 14 CSS files are loaded by `index.html`.
- 15 CSS files exist, totaling about 3,573 lines.
- Two independent token systems exist (`base.css` and `ui-system.css`).
- 72 selectors are defined in more than one CSS file in the current source tree.
- `ui-system.css` is ~1,025 lines and contains 48 `!important` declarations, mostly overriding older stage styles.
- UI behavior files also create their own layout DOM: `editor.js` ~1,296 lines, `review.js` ~697, `preview.js` ~458.
- Page jump/navigation is implemented independently in Preview, Review, and Editor.
- `review-workspace.js` wraps the legacy Review renderer and removes/re-homes legacy controls after render. This is migration scaffolding, not a sustainable final architecture.
- `box-item.js`, `editor-properties.js`, `editor-workspace.js`, and `editor-properties.css` are not loaded by `index.html`; the architecture document already identifies the three JS files as legacy.
- Existing UI tests partially lock the migration scaffolding itself (four horizontal workflow buttons, renderer wrappers, `legacyToolbar?.remove()`), rather than only protecting user behavior.

## Target information architecture

Desktop workbench:

```text
┌──────────┬──────────────────┬────────────────────────────────┬──────────────────────┐
│ App rail │ Page navigator   │ Canvas / primary workspace     │ Context inspector    │
│          │                  │                                │                      │
│ Import   │ page thumbnails  │ current manga slice/page       │ selected page/region │
│ Process  │ state/filter     │ overlays + direct manipulation │ OCR / translation    │
│ Review   │ search/jump      │ compact contextual tools       │ style / QC / history │
│ Editor   │                  │                                │                      │
│          │                  │                                │                      │
│ Settings │                  │                                │                      │
└──────────┴──────────────────┴────────────────────────────────┴──────────────────────┘
```

A thin top command bar may contain chapter identity, current stage, save/job state, zoom, and one primary action. It must not become a second navigation system.

Landing is the exception: it can use a focused start screen before a chapter is opened.

## Keep / rework / replace / remove

### Keep concept, redesign implementation

- Stage model: Import → Process → Review → Editor.
- Global Settings as the home for provider/API configuration.
- Manifest-backed resume/checkpoint behavior.
- Canvas overlays and direct selection.
- Review brush/magic-wand/repaint behavior.
- Editor text-object manipulation and render/export behavior.
- Semantic state colors for verified/review/error/rendered/skipped.

### Rework

- Landing: keep URL/local import, but collapse visual hierarchy and make recent chapters first-class rather than a secondary card appended below.
- Settings: keep drawer/modal behavior, replace provider-specific ad-hoc blocks with consistent settings sections.
- Status: retain semantic color but add icon/text; never encode critical state by color only.
- Toolbars: reduce to contextual commands; long-lived configuration goes to inspector/settings.

### Replace

1. Horizontal `workflow-steps` header → vertical App Rail.
2. Per-stage Prev/Next/jump navigation → one shared Page Navigator.
3. Preview bottom thumbnail strip → Page Navigator thumbnails/state.
4. Preview canvas inside a decorative `preview-card` → direct workbench canvas surface.
5. Review toolbar packed with brush/clear/AI/repaint/reset/help → compact canvas tool strip + contextual Review inspector.
6. `chapter-qc-panel` below/around the canvas → AI/QC inspector section with job progress and findings list.
7. Editor panel/card stack → selection-driven Context Inspector.
8. Horizontal region chip/list workaround → Region list/tab inside inspector or page navigator context.
9. CSS override stack → one token layer + workbench primitives + small stage modules.
10. `ui-shell.js` renderer-wrapping migration mechanism → explicit workbench state/router API.
11. DOM-layout construction duplicated across stage behavior files → shared shell/components; stage modules only provide task content/actions.

### Remove

Safe immediate legacy removal:
- `app/static/js/box-item.js`
- `app/static/js/editor-properties.js`
- `app/static/js/editor-workspace.js`
- `app/static/css/editor-properties.css`

Remove later, only after replacements are active:
- Legacy horizontal workflow header styles.
- Duplicate page navigation DOM/CSS in Preview/Review/Editor.
- Preview thumbnail strip.
- Legacy editor `box-panel` presentation layer.
- Review legacy-toolbar migration code.
- Compatibility token aliases (`--ink`, `--panel`, `--paper`, `--blue`, etc.).
- `ui-system.css` override rules that exist only to beat earlier stage CSS.

## Design constitution

1. Canvas first. Navigation and chrome must be visually quieter than manga content.
2. Context follows selection. Inspector content changes with the selected page/region/object.
3. One persistent navigation model. Never implement stage-specific page navigation again.
4. One primary action per task context.
5. Configuration is not a toolbar. Persistent provider/model/preferences belong in Settings/Inspector.
6. No card nesting for layout. Cards are for bounded data objects, not every container.
7. Four surface levels maximum: app, panel, raised interaction, overlay/popover.
8. Use one spacing scale: 4/8/12/16/24/32.
9. Use one radius scale and one control-height scale.
10. No decorative gradients/glow unless they communicate state or depth.
11. Semantic colors are reserved for state/action meaning.
12. Every icon-only action requires a tooltip and accessible label.
13. Keyboard focus must always be visible; repetitive editing actions should gain shortcuts.
14. Minimum target size follows WCAG guidance; compact controls may be visually small but need adequate hit area.
15. Motion is short and functional. Avoid long transitions on repetitive editing operations.
16. UI must be usable at 1280px desktop without hiding the canvas behind panels.
17. Side panels should be collapsible; selecting an object may reveal the inspector automatically.
18. Mobile is a task-preserving fallback, not a squeezed desktop layout: panels become drawers/sheets.
19. No `!important` in new component CSS except documented third-party/browser normalization.
20. A visual redesign is not complete until old conflicting CSS/DOM is deleted.

## Component inventory for v0.3

Shared primitives:
- `WorkbenchShell`
- `AppRail`
- `TopCommandBar`
- `PageNavigator`
- `CanvasViewport`
- `CanvasToolStrip`
- `ContextInspector`
- `InspectorSection`
- `StatusBadge`
- `JobProgress`
- `CommandButton` / `IconButton`
- `Field` / `Select` / `Slider` / `ColorControl`
- `Toast`
- `SettingsSurface`

Stage adapters:
- ImportStart
- ProcessInspector
- ReviewInspector
- TranslationInspector

The shared components own layout. Stage modules own domain behavior.

## Screen migration

### Landing / Import

Keep:
- URL import
- file/ZIP/CBZ import
- language and worker controls
- recent chapters

Change:
- One focused start area, not two equally loud large cards separated by a vertical “or”.
- Recent chapters become a clear resume list.
- Advanced processing controls can live in an expandable Advanced section.

### Process / Preview

Remove the canvas card chrome and bottom thumbnail strip.

New layout:
- Left: pages/slices with skipped/excluded state.
- Center: image canvas with zoom and exclusion overlays.
- Right: page processing inspector (skip, exclusions, slice/source metadata).
- Top primary action: Process selected / Process chapter.

### Review

New layout:
- Left: pages with verified/review/error counts.
- Center: clean image + brush/highlight overlays.
- Canvas tool strip: select/brush/clear as gesture tools only.
- Right inspector: current page issues, repaint/reset actions, AI QC provider/job/findings.

The current chapter QC panel and Help block do not stay as standalone canvas-adjacent cards.

### Editor

New layout:
- Left: pages and optional region list/filter.
- Center: translated canvas and direct manipulation.
- Right: selected region inspector with OCR, translation, geometry, typography, stroke/background, duplicate/delete.
- Top: Add region, render page/chapter, save state, export.

The old `box-panel` list/card presentation is replaced by the selection-driven inspector.

## CSS migration

Target ownership:

- `tokens.css`: colors, typography, spacing, radii, z-index, motion.
- `workbench.css`: rail/sidebar/topbar/canvas/inspector geometry and responsive behavior.
- `components.css`: buttons, fields, badges, tooltips, progress, toasts, settings.
- `canvas.css`: canvas/overlay/direct-manipulation primitives.
- `stages/*.css`: only truly stage-specific visual rules.

During migration, old CSS may coexist temporarily, but every migrated component must delete the rules it replaces. The goal is fewer loaded stylesheets and zero override-layer dependence.

## JS migration

Create shared navigation/workbench state first. Do not rewrite domain APIs.

- Centralize page index/jump/Prev/Next logic.
- Centralize shell mounting/unmounting and panel slots.
- Keep manifest and endpoint semantics unchanged.
- Preserve pending-save flush and busy locks when switching stages/pages.
- Move DOM layout creation out of `preview.js`, `review*.js`, and `editor.js` incrementally.

## Test migration

Preserve behavior tests for:
- stage access rules and checkpoints
- page index continuity between stages
- busy-state navigation locks
- stale-result rejection
- OCR/QC job state
- render/export correctness

Replace layout-coupled assertions that require:
- four horizontal workflow buttons
- `wrapRenderer(...)`
- `legacyToolbar?.remove()`
- old toolbar/panel class names

New UI contract tests should assert workbench landmarks, accessible navigation, inspector behavior, shared page navigation, focus/keyboard behavior, and absence of document horizontal overflow.

## Implementation order

1. Phase 0 — audit/constitution + remove definitely dead modules.
2. Phase 1 — tokens + WorkbenchShell/AppRail/PageNavigator/Inspector skeleton; preserve APIs.
3. Phase 2 — migrate Landing and Process.
4. Phase 3 — migrate Review + AI/QC into inspector.
5. Phase 4 — migrate Editor + text properties into inspector.
6. Phase 5 — delete legacy CSS/DOM, update tests, responsive/accessibility browser gate.
7. Only after visual/behavior closure: merge v0.3 UI branch.
