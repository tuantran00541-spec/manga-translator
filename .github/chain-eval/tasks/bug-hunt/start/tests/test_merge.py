from ranges import merge_ranges, total_length


def test_overlap():
    assert merge_ranges([[1, 3], [2, 5]]) == [[1, 5]]


def test_unsorted():
    assert merge_ranges([[8, 9], [1, 2]]) == [[1, 2], [8, 9]]


def test_total_counts_overlap_once():
    assert total_length([[1, 4], [2, 6]]) == 5
