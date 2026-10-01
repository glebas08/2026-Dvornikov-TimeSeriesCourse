import numpy as np


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """

    ts1 = np.asarray(ts1, dtype=float)
    ts2 = np.asarray(ts2, dtype=float)

    ed_dist = np.sqrt(np.sum((ts1 - ts2) ** 2))

    return ed_dist


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """

    ts1 = np.asarray(ts1, dtype=float)
    ts2 = np.asarray(ts2, dtype=float)
    n = ts1.shape[0]

    mu1, mu2 = np.mean(ts1), np.mean(ts2)
    sigma1, sigma2 = np.std(ts1), np.std(ts2)
    dot_product = np.dot(ts1, ts2)

    norm_ed_dist = np.sqrt(np.abs(2 * n * (1 - (dot_product - n * mu1 * mu2) / (n * sigma1 * sigma2))))

    return norm_ed_dist


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    ts1 = np.asarray(ts1, dtype=float)
    ts2 = np.asarray(ts2, dtype=float)
    n, m = len(ts1), len(ts2)

    # ширина окна Сакоэ-Чиба в отсчётах (r = 1 -> окно на всю матрицу)
    w = max(int(np.ceil(r * max(n, m))), abs(n - m))

    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(max(1, i - w), min(m, i + w) + 1):
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2
            d[i, j] = cost + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])

    dtw_dist = d[n, m]

    return dtw_dist
