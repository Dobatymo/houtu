import numpy as np


def rand_lat_lon(samples: int, form: str = "radians") -> np.ndarray:
    arr = np.random.rand(samples, 2).astype(np.float32) - 0.5

    if form == "radians":
        arr[:, 0] *= np.pi / 2.0
        arr[:, 1] *= np.pi
    elif form == "degrees":
        arr[:, 0] *= 90.0
        arr[:, 1] *= 180.0
    else:
        raise ValueError(f"Invalid input form: {form}")

    return arr
