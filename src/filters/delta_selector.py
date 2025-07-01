from typing import Literal
from framework3.base.base_clases import BaseFilter
from framework3.base.base_types import XYData
from framework3.container import Container

from scipy.sparse import dok_matrix
import numpy as np


@Container.bind()
class DeltaSelectorFilter(BaseFilter):
    def __init__(
        self,
        deltas_f: Literal[
            "cosine",
            "euclidean",
            "chebyshev",
            "jensen_shannon",
            "wasserstein",
            "manhattan",
            "minkowski",
        ] = "cosine",
    ):
        self.deltas_f = deltas_f

    def fit(self, x: XYData, y: XYData | None):
        pass

    def predict(self, x: XYData) -> XYData:
        # Crear una nueva dok_matrix con el mismo shape y en float32
        old_dok = x.value[self.deltas_f]
        new_dok = dok_matrix(old_dok.shape, dtype=np.float32)

        # Copiar todos los valores existentes y convertir el dtype
        for (i, j), value in old_dok.items():
            new_dok[i, j] = float(value)  # conversión a float32 implícita
        return XYData.mock(new_dok.tocsr())
