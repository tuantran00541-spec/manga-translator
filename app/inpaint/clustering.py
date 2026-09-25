
from app.detector.bubble_detector import BubbleBox
from app.parameters import (
    DETECTOR_MAX_BOX_AREA_RATIO as MAX_BOX_AREA_RATIO,
    INPAINT_CLUSTER_MAX_DIM,
    INPAINT_CLUSTER_GROUP_HEIGHT_FACTOR,
    INPAINT_CLUSTER_LINE_OVERLAP_MIN,
    INPAINT_CLUSTER_PADDING,
    INPAINT_CLUSTER_SPLIT_COUNT,
    INPAINT_CLUSTER_SPLIT_HEIGHT_FACTOR,
    INPAINT_CROP_LONG_ASPECT_THRESHOLD,
    INPAINT_CROP_PADDING,
    MANUAL_CROP_PADDING,
)


def cluster_boxes(boxes: list[BubbleBox]) -> list[list[BubbleBox]]:
    remaining = list(boxes)
    raw_clusters = []

    while remaining:
        current = [remaining.pop(0)]
        changed = True
        while changed:
            changed = False
            still_remaining = []
            for b in remaining:
                if any(boxes_close(b, c) for c in current) and can_add_to_cluster(current, b, INPAINT_CLUSTER_MAX_DIM):
                    current.append(b)
                    changed = True
                else:
                    still_remaining.append(b)
            remaining = still_remaining
        raw_clusters.append(current)

    final_clusters = []
    for cluster in raw_clusters:
        if len(cluster) > 1:
            avg_h = sum(b.y2 - b.y1 for b in cluster) / len(cluster)
            cluster_h = max(b.y2 for b in cluster) - min(b.y1 for b in cluster)
            if (
                len(cluster) > INPAINT_CLUSTER_SPLIT_COUNT
                or cluster_h > INPAINT_CLUSTER_SPLIT_HEIGHT_FACTOR * avg_h
            ):
                sub_clusters = split_cluster_lines(cluster, avg_h)
                final_clusters.extend(sub_clusters)
            else:
                final_clusters.append(cluster)
        else:
            final_clusters.append(cluster)

    return final_clusters



def split_oversized_cluster_area(
    cluster: list[BubbleBox],
    img_w: int,
    img_h: int,
) -> list[list[BubbleBox]]:
    if not cluster:
        return []
    page_limit = max(1.0, float(img_w * img_h) * MAX_BOX_AREA_RATIO)
    pending = [list(cluster)]
    result: list[list[BubbleBox]] = []
    while pending:
        group = pending.pop()
        x1 = min(b.x1 for b in group)
        y1 = min(b.y1 for b in group)
        x2 = max(b.x2 for b in group)
        y2 = max(b.y2 for b in group)
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if len(group) <= 1 or area <= page_limit:
            result.append(group)
            continue

        span_x = x2 - x1
        span_y = y2 - y1
        if span_x >= span_y:
            ordered = sorted(group, key=lambda b: ((b.x1 + b.x2), b.y1, b.x1))
        else:
            ordered = sorted(group, key=lambda b: ((b.y1 + b.y2), b.x1, b.y1))
        midpoint = max(1, len(ordered) // 2)
        left = ordered[:midpoint]
        right = ordered[midpoint:]
        if not right:
            result.extend([[b] for b in ordered])
            continue
        pending.append(right)
        pending.append(left)

    result.sort(
        key=lambda group: (
            min(b.y1 for b in group),
            min(b.x1 for b in group),
        )
    )
    return result



def split_cluster_lines(cluster: list[BubbleBox], avg_h: float) -> list[list[BubbleBox]]:
    sorted_boxes = sorted(cluster, key=lambda b: (b.y1, b.x1))
    lines = []
    for b in sorted_boxes:
        placed = False
        for line in lines:
            line_y1 = min(x.y1 for x in line)
            line_y2 = max(x.y2 for x in line)
            overlap = min(b.y2, line_y2) - max(b.y1, line_y1)
            min_h = min(b.y2 - b.y1, line_y2 - line_y1)
            if (
                min_h > 0
                and overlap / min_h > INPAINT_CLUSTER_LINE_OVERLAP_MIN
            ):
                line.append(b)
                placed = True
                break
        if not placed:
            lines.append([b])

    lines.sort(key=lambda line: min(b.y1 for b in line))
    sub_clusters = []
    current_group = []
    for line in lines:
        if not current_group:
            current_group = list(line)
        else:
            group_h = max(b.y2 for b in current_group + line) - min(b.y1 for b in current_group + line)
            if group_h > INPAINT_CLUSTER_GROUP_HEIGHT_FACTOR * avg_h:
                sub_clusters.append(current_group)
                current_group = list(line)
            else:
                current_group.extend(line)
    if current_group:
        sub_clusters.append(current_group)

    return sub_clusters



def boxes_close(a: BubbleBox, b: BubbleBox) -> bool:
    ax1, ay1, ax2, ay2 = (
        a.x1 - INPAINT_CLUSTER_PADDING,
        a.y1 - INPAINT_CLUSTER_PADDING,
        a.x2 + INPAINT_CLUSTER_PADDING,
        a.y2 + INPAINT_CLUSTER_PADDING,
    )
    bx1, by1, bx2, by2 = b.x1, b.y1, b.x2, b.y2
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)



def can_add_to_cluster(
    cluster: list[BubbleBox],
    b: BubbleBox,
    max_dim: int = INPAINT_CLUSTER_MAX_DIM,
) -> bool:
    x1 = min(min(box.x1 for box in cluster), b.x1)
    y1 = min(min(box.y1 for box in cluster), b.y1)
    x2 = max(max(box.x2 for box in cluster), b.x2)
    y2 = max(max(box.y2 for box in cluster), b.y2)
    return (x2 - x1) <= max_dim and (y2 - y1) <= max_dim



def compute_manual_crop_region(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> tuple:
    x1 = max(0, x1 - MANUAL_CROP_PADDING)
    y1 = max(0, y1 - MANUAL_CROP_PADDING)
    x2 = min(img_w, x2 + MANUAL_CROP_PADDING)
    y2 = min(img_h, y2 + MANUAL_CROP_PADDING)
    return int(x1), int(y1), int(x2), int(y2)



def compute_crop_region(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> tuple:
    x1 -= INPAINT_CROP_PADDING
    y1 -= INPAINT_CROP_PADDING
    x2 += INPAINT_CROP_PADDING
    y2 += INPAINT_CROP_PADDING

    box_w = x2 - x1
    box_h = y2 - y1

    aspect = max(box_w / max(1, box_h), box_h / max(1, box_w))
    if aspect > INPAINT_CROP_LONG_ASPECT_THRESHOLD:
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)
        return int(x1), int(y1), int(x2), int(y2)

    side = max(box_w, box_h)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    x1 = cx - side / 2
    x2 = cx + side / 2
    y1 = cy - side / 2
    y2 = cy + side / 2

    if x1 < 0:
        x2 = min(img_w, x2 - x1)
        x1 = 0
    if y1 < 0:
        y2 = min(img_h, y2 - y1)
        y1 = 0
    if x2 > img_w:
        x1 = max(0, x1 - (x2 - img_w))
        x2 = img_w
    if y2 > img_h:
        y1 = max(0, y1 - (y2 - img_h))
        y2 = img_h

    return int(x1), int(y1), int(x2), int(y2)
