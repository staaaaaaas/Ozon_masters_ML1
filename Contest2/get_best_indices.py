import numpy as np


def get_best_indices(rank, top, axis=1):
    return np.apply_along_axis(lambda x: np.flip(np.argpartition(x, np.arange(
        x.shape[0] - top, x.shape[0]))[-top:]), axis=axis, arr=rank)


if __name__ == "__main__":
    with open('input.bin', 'rb') as f_data:
        ranks = np.load(f_data)
    indices = get_best_indices(rank=ranks, top=5)
    with open('output.bin', 'wb') as f_data:
        np.save(f_data, indices)
