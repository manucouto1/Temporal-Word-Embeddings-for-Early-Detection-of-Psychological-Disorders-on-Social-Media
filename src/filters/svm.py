import itertools
from typing import Any, Callable, Literal, Mapping, cast
from sklearn.svm import SVC

from framework3 import Container, XYData
from framework3.base import BaseFilter


@Container.bind()
class ClassifierSVM(BaseFilter):
    def __init__(
        self,
        C: float = 1,
        kernel: Callable
        | Literal["linear", "poly", "rbf", "sigmoid", "precomputed"] = "rbf",
        gamma: float | Literal["scale", "auto"] = "scale",
        coef0: float = 0.0,
        tol: float = 0.001,
        decision_function_shape: Literal["ovo", "ovr"] = "ovr",
        class_weight_1: Mapping[Any, Any] | str | None = None,
        probability: bool = False,
    ):
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.decision_function_shape = decision_function_shape
        self.class_weight_1 = class_weight_1
        self.probability = probability

        self._model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            decision_function_shape=decision_function_shape,
            class_weight={int(k): v for k, v in cast(dict, class_weight_1).items()}
            if class_weight_1 is not None
            else None,
            probability=probability,
            random_state=43,
        )

    def fit(self, x: XYData, y: XYData | None):
        if y is None:
            raise ValueError("y must be provided for training")

        yy = list(map(lambda pair: [pair[1]] * len(pair[0]), zip(x.value, y.value)))

        yy = list(itertools.chain.from_iterable(yy))

        xx = x.value.reshape(-1, x.value.shape[-1]).numpy()
        self._model.fit(xx, yy)

    def predict(self, x: XYData) -> XYData:
        xx = x.value.reshape(-1, x.value.shape[-1]).numpy()
        if self.probability:
            result = list(map(lambda i: i[1], self._model.predict_proba(xx)))
            return XYData.mock(result)
        else:
            result = self._model.predict(xx)
            return XYData.mock(result)
