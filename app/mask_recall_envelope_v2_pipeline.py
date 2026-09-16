from __future__ import annotations

from app.mask_recall_envelope_pipeline import FlatEnvelopeMaskRecallPipeline


class CompleteFlatEnvelopeMaskRecallPipeline(FlatEnvelopeMaskRecallPipeline):
    """Finish flat-container edge glyphs clipped just beyond the V2 envelope.

    Real-page audit on Shadow Slave p117 showed the source detector box ending at
    x=1266, while the final surviving glyph fragment occupied roughly x=1362-1365.
    The previous 96 px horizontal envelope therefore ended exactly through the
    glyph. Keep every flat-container/ring/component safety gate from V2 and add
    only enough horizontal slack to contain that observed failure mode.
    """

    _FLAT_SEARCH_PAD_X = 128
    _FLAT_SEARCH_PAD_Y = 32
