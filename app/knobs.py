"""Every number that decides how text is found, erased and filled, in one place with the range it may be tuned in."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Knob:
    name: str
    default: float | int
    low: float | int
    high: float | int
    doc: str

    @property
    def stage(self) -> str:
        """``detect``, ``mask`` or ``fill``: the step the knob belongs to."""
        return self.name.split(".", 1)[0]


_ALL = (
    # Finding text: the Kiuyha detector and how its boxes are joined and parted.
    Knob("detect.confidence", 0.25, 0.10, 0.60, "detector score a text box needs"),
    Knob("detect.nms_iou", 0.7, 0.4, 0.9, "overlap above which two detections are one (Ultralytics' default)"),
    Knob("detect.box_pad", 16, 0, 48, "pixels added round a box, as Kiuyha can stop short of a wide line's last letter"),
    Knob("detect.edge_touch", 3, 0, 12, "a box this close to an inner band edge was cut by it"),
    Knob("detect.union_share", 0.3, 0.1, 0.9, "boxes sharing this much of the smaller one are one text"),
    Knob("detect.same_line", 0.7, 0.4, 0.95, "overlapping boxes sharing this much of the shorter height are one line"),
    Knob("detect.block_gap", 1.5, 0.8, 3.0, "empty band, in line heights, that separates two texts"),
    Knob("detect.block_offset", 0.15, 0.05, 0.4, "centres this far apart, as a share of the wider text, are two texts"),
    Knob("detect.block_spacing", 3.0, 1.5, 6.0, "a gap this many times the widest inner line gap parts two texts"),
    Knob("detect.block_min_width", 0.4, 0.1, 0.8, "a group narrower than this share of the other is a stray mark"),
    Knob("detect.block_line_spread", 2.0, 1.3, 4.0, "a row this many times taller than a line is a mark, not text"),
    # The erase mask: letters from comic-text-detector, grown into outline, glow and shadow.
    Knob("mask.ink_threshold", 0.3, 0.15, 0.6, "letter probability that counts as ink"),
    Knob("mask.pad_share", 0.25, 0.0, 0.6, "room round a box, as a share of its height, for letters its edge cuts"),
    Knob("mask.reach", 1.0, 0.3, 2.0, "how far, in letter heights, outline and glow may spread from the letters"),
    Knob("mask.outline", 0.12, 0.0, 0.4, "share of the reach every letter keeps as outline"),
    Knob("mask.edge", 60.0, 20.0, 120.0, "Lab lightness gradient above which a pixel is drawn art, not glow"),
    Knob("mask.ring", 6, 2, 16, "width of the band past the crop where background colours are read"),
    Knob("mask.left_share", 0.004, 0.001, 0.02, "share of an erased block still read as text before another pass"),
    Knob("mask.reach_rounds", 3, 0, 6, "times a box grows toward letters its edge still cuts"),
    Knob("mask.chain", 0.6, 0.2, 1.5, "widest gap, in letter heights, between letters of one text"),
    Knob("mask.fringe", 0.5, 0.0, 1.0, "how far, in letter heights, a soft shadow or blur may fade from the letters"),
    Knob("mask.fringe_tolerance", 5.0, 2.0, 12.0, "Lab distance within which a pixel is the flat background"),
    Knob("mask.fringe_flat_share", 0.6, 0.3, 0.9, "share of the band round the letters that must be that background"),
    Knob("mask.fringe_ink_share", 0.8, 0.5, 0.95, "a shadow pixel stays this much lighter than the ink"),
    Knob("mask.speck", 0.35, 0.15, 0.6, "letters shorter than this share of the text size never lead growth"),
    Knob("mask.speck_hops", 3, 0, 6, "specks past the letter a run of specks may reach"),
    Knob("mask.dot", 0.15, 0.05, 0.3, "smallest side of a dot in a run of dots, as a share of the text size"),
    # Filling the hole: flat fill when the ring is one colour, otherwise LaMa on a crop round the cluster.
    Knob("fill.lama_side", 512, 256, 1280, "longest side LaMa fills a crop at"),
    Knob("fill.crop_padding", 48, 16, 128, "context round a cluster given to LaMa"),
    Knob("fill.cluster_padding", 35, 0, 96, "holes this close are filled in one LaMa crop"),
    Knob("fill.cluster_max_dim", 600, 256, 1200, "largest cluster before it is split"),
    Knob("fill.ring_margin", 6, 2, 16, "width of the ring round a hole read to choose a flat fill"),
    Knob("fill.ring_near_px", 3, 1, 8, "pixels right round a hole that must share one colour for a flat fill"),
    Knob("fill.ring_agree", 0.98, 0.9, 1.0, "share of those pixels within the tolerance of the ring's colour"),
    Knob("fill.ring_tolerance", 12.0, 4.0, 24.0, "Lab distance from the ring's colour that still counts as the same"),
    Knob("fill.white_std_max", 8.0, 3.0, 16.0, "spread of a white ring that is still flat paper"),
    Knob("fill.full_std_max", 12.0, 5.0, 24.0, "spread of the wider context that still allows a flat fill"),
    Knob("fill.edge_density_max", 0.01, 0.002, 0.05, "share of edge pixels in the context that still allows a flat fill"),
    Knob("fill.chroma_std_max", 12.0, 4.0, 24.0, "colour spread of the ring and context that still allows a flat fill"),
)

KNOBS: dict[str, Knob] = {knob.name: knob for knob in _ALL}


def knob(name: str) -> float | int:
    """The default value of a knob."""
    return KNOBS[name].default

