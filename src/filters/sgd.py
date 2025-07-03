from typing import Literal
from sklearn.linear_model import SGDClassifier

from framework3 import Container, XYData
from framework3.base import BaseFilter


@Container.bind()
class ClassifierSGD(BaseFilter):
    def __init__(
        self,
        loss: Literal[
            "hinge",
            "log_loss",
            "log",
            "modified_huber",
            "squared_hinge",
            "perceptron",
            "squared_error",
            "huber",
            "epsilon_insensitive",
            "squared_epsilon_insensitive",
        ] = "hinge",  # SVM lineal
        penalty: Literal["l2", "l1", "elasticnet"]
        | None = "l2",  # Regularización Ridge
        alpha=1e-4,  # Regularización débil
        l1_ratio: float = 0.15,
        max_iter=1000,
        tol=1e-3,
        early_stopping=True,
        class_weight="balanced",
    ):
        super().__init__()
        self._model = SGDClassifier(
            loss=loss,  # SVM lineal
            penalty=penalty,  # Regularización Ridge
            alpha=alpha,  # Regularización débil
            max_iter=max_iter,
            tol=tol,
            l1_ratio=l1_ratio,
            early_stopping=early_stopping,
            class_weight=class_weight,
        )

    def fit(self, x: XYData, y: XYData | None):
        if y is None:
            raise ValueError("y must be provided for training")
        self._model.fit(x.value, y.value)

    def predict(self, x: XYData) -> XYData:
        result = self._model.predict(x.value)
        return XYData.mock(result)
