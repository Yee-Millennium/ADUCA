def construct_block_range(begin: int, end: int, block_size):
    blocks = []

    for start in range(begin, end, block_size):
        stop = min(start + block_size, end)  # Ensure the end doesn't exceed total columns
        block_range = range(start, stop)
        blocks.append(block_range)
    return blocks