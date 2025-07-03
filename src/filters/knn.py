
from typing import Callable, Literal
from sklearn.neighbors import KNeighborsClassifier

from framework3 import Container, XYData
from framework3.base import BaseFilter


@Container.bind()
class ClassifierKNN(BaseFilter):
    def __init__(
        self, 
        n_neighbors: int = 5,
        weights:  Callable| Literal['uniform', 'distance'] | None = "uniform",
        algorithm: Literal['auto', 'ball_tree', 'kd_tree', 'brute'] = "auto",
        leaf_size: int = 30,
        p: int = 2,
        metric: str | Callable = "minkowski",
        probability:bool=True,
    ):
        super().__init__()
        self.probability = probability
        self._model = KNeighborsClassifier(
            n_neighbors=n_neighbors, 
            weights=weights, 
            algorithm=algorithm, 
            leaf_size=leaf_size, 
            p=p, 
            metric=metric
        )
        
    def fit(self, x, y):
        self._model.fit(x.value, y.value)
    
    def predict(self, x ):
        if self.probability:
            return XYData.mock(self._model.predict_proba(x.value))
        return XYData.mock(self._model.predict(x.value))