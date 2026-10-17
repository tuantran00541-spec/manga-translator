def merge_ranges(ranges):
    """Merge [start, end] pairs that overlap or touch; return them sorted."""
    out = []
    for start, end in ranges:
        if out and start < out[-1][1]:
            out[-1][1] = end
        else:
            out.append([start, end])
    return out


def total_length(ranges):
    """Total length covered by the ranges, counting overlaps once."""
    return sum(end - start for start, end in ranges)
