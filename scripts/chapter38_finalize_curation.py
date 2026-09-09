from __future__ import annotations

import json
from pathlib import Path
from PIL import Image

ROOT = Path('chapter38-ocr-review')
MANIFEST = ROOT / 'processed' / 'manifest.json'
DROP_REASON = 'drop_duplicate_credit_sfx_or_noise'

manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))

# Human-reviewed detector rows marked as duplicate/credit/SFX/noise must remain
# tombstoned after production manifest reconciliation. Auto-generated objects
# otherwise re-sync to their still-active source box and clear source_missing.
tombstoned = 0
for page in manifest.get('pages', []):
    for obj in page.get('text_objects') or []:
        if obj.get('human_review') != DROP_REASON:
            continue
        obj['source_missing'] = True
        obj['translation'] = ''
        obj['auto_generated'] = False
        obj['origin'] = 'human_review_drop'
        tombstoned += 1

# The detector/inpaint pass erased the non-story end-card credits and left a
# visibly smeared field. Credits/watermarks are not dialogue, so preserve the
# source artwork rather than pretending they are untranslated story text.
raw_end = ROOT / 'raw' / 'sliced' / '023_05.png'
clean_end = ROOT / 'processed' / 'clean_023_05.png'
if raw_end.is_file() and clean_end.is_file():
    raw = Image.open(raw_end).convert('RGB')
    clean = Image.open(clean_end).convert('RGB')
    y1 = min(3000, raw.height)
    if raw.size != clean.size:
        raise SystemExit(f'end-card size mismatch: raw={raw.size}, clean={clean.size}')
    clean.paste(raw.crop((0, y1, raw.width, raw.height)), (0, y1))
    clean.save(clean_end)
    raw.close(); clean.close()
else:
    raise SystemExit('missing raw/clean final slice for end-card restoration')

MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

summary_path = ROOT / 'chapter38-curation-summary.json'
summary = json.loads(summary_path.read_text(encoding='utf-8'))
summary['persistent_human_drop_tombstones'] = tombstoned
summary['end_card_credit_restored_from_raw'] = True
summary['finalization_note'] = 'all story dialogue/free-text translated; scanlation/production credits preserved as source credits'
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

active_untranslated = []
for pi, page in enumerate(manifest.get('pages', [])):
    for obj in page.get('text_objects') or []:
        if obj.get('source_missing'):
            continue
        src = str(obj.get('ocr_text') or '').strip()
        vi = str(obj.get('translation') or '').strip()
        if src and not vi:
            active_untranslated.append({'page_index': pi, 'id': obj.get('id'), 'source': src})

if active_untranslated:
    raise SystemExit(f'active untranslated rows remain after final curation: {active_untranslated[:10]}')

print(json.dumps({
    'tombstoned_human_drops': tombstoned,
    'active_untranslated': len(active_untranslated),
    'end_card_restored': True,
}, ensure_ascii=False, indent=2))
