from sklearn.ensemble import RandomForestClassifier
from typing import Literal, Mapping, Sequence
from sklearn.linear_model import SGDClassifier

from framework3 import Container, XYData
from framework3.base import BaseFilter


@Container.bind()
class ClassifierRFC(BaseFilter):
    def __init__(
        self, 
        n_estimators: int = 100,
        criterion: Literal['gini', 'entropy', 'log_loss'] = "gini",
        max_depth: int | None = None,
        min_samples_split: float = 2,
        min_samples_leaf: float = 1,
        max_features: float | Literal['sqrt', 'log2'] = "sqrt",
        class_weight: Mapping | Sequence[Mapping] | Literal['balanced', 'balanced_subsample'] | None = None,
        proba: bool = False
    ):
        super().__init__()
        self.proba = proba
        self._clf = RandomForestClassifier(
            n_estimators=n_estimators, 
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,            
            random_state=0)
        
    def fit(self, x, y=None):
        self._clf.fit(x.value, y.value)
    
    def predict(self, x, y=None):
        if self.proba:
            return list(map(lambda i: i[1], self._clf.predict_proba(x.value)))
        return XYData.mock(self._clf.predict(x.value))
