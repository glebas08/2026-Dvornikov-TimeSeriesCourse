import numpy as np

from modules.metrics import ED_distance, norm_ED_distance, DTW_distance
from modules.utils import z_normalize


class PairwiseDistance:
    """
    Distance matrix between time series 

    Parameters
    ----------
    metric: distance metric between two time series
            Options: {euclidean, dtw}
    is_normalize: normalize or not time series
    """

    def __init__(self, metric: str = 'euclidean', is_normalize: bool = False) -> None:

        self.metric: str = metric
        self.is_normalize: bool = is_normalize
    

    @property
    def distance_metric(self) -> str:
        """Return the distance metric

        Returns
        -------
            string with metric which is used to calculate distances between set of time series
        """

        norm_str = ""
        if (self.is_normalize):
            norm_str = "normalized "
        else:
            norm_str = "non-normalized "

        return norm_str + self.metric + " distance"


    def _choose_distance(self):
        """ Choose distance function for calculation of matrix
        
        Returns
        -------
        dict_func: function reference
        """

        if self.metric == 'euclidean':
            # для нормализованных рядов нормализация "встроена" в саму формулу
            dist_func = norm_ED_distance if self.is_normalize else ED_distance
        elif self.metric == 'dtw':
            # для DTW нормализация выполняется заранее, в calculate(),
            # поэтому здесь всегда используется обычная DTW_distance
            dist_func = DTW_distance
        else:
            raise ValueError(f"Unknown metric '{self.metric}'. Options: 'euclidean', 'dtw'")

        return dist_func


    def calculate(self, input_data: np.ndarray) -> np.ndarray:
        """ Calculate distance matrix
        
        Parameters
        ----------
        input_data: time series set
        
        Returns
        -------
        matrix_values: distance matrix
        """
        
        matrix_shape = (input_data.shape[0], input_data.shape[0])
        matrix_values = np.zeros(shape=matrix_shape)

        input_data = np.asarray(input_data, dtype=float)
        K = input_data.shape[0]
        matrix_values = np.zeros(shape=(K, K))

        # для всех мер, кроме евклидовой, нормализация делается заранее через z_normalize
        if self.is_normalize and self.metric != 'euclidean':
            input_data = np.array([z_normalize(ts) for ts in input_data])

        dist_func = self._choose_distance()

        for i in range(K):
            for j in range(i + 1, K):
                dist = dist_func(input_data[i], input_data[j])
                matrix_values[i, j] = dist
                matrix_values[j, i] = dist

        return matrix_values
