from sklearn.metrics import confusion_matrix
from framework3 import BaseMetric, Container, XYData

from numpy import exp
import numpy as np

__all__ = ["ERDE_5", "ERDE_50"]


class ERDE(BaseMetric):
    def __init__(self, k: int = 5):
        self.k = k

    @staticmethod
    def __textos_antes_de_y_1(grupo):
        grupo = grupo.reset_index(drop=True)
        try:
            idx = grupo["yy"].tolist().index(1)
            return grupo.loc[:idx, "n_texts"].astype(int).sum()
        except ValueError:
            return grupo["n_texts"].sum()

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> float | np.ndarray:
        if y_true is None:
            raise ValueError("y_true must be provided for evaluation")

        x_df = x_data.value.copy()
        x_df.loc[:, "label"] = y_true.value
        x_df = x_df.explode(["chunk", "text", "date", "n_texts"]).reset_index(drop=True)
        x_df["yy"] = y_pred.value

        acum_texts = x_df.groupby(["user"]).apply(ERDE.__textos_antes_de_y_1)

        aux = x_df.groupby(["user"]).agg(
            {"label": "first", "yy": lambda x: 1 if any(x) else 0}
        )
        aux["n_texts"] = acum_texts.values

        all_erde = []
        _, _, _, tp = confusion_matrix(x_df.label.values, x_df.yy.values).ravel()
        for row in aux.itertuples():
            expected, result, count = row.label, row.yy, row.n_texts
            if result == 1 and expected == 0:
                all_erde.append(float(tp) / len(x_df.label.values))
            elif result == 0 and expected == 1:
                all_erde.append(1.0)
            elif result == 1 and expected == 1:
                all_erde.append(1.0 - (1.0 / (1.0 + exp(count - self.k))))
            elif result == 0 and expected == 0:
                all_erde.append(0.0)
        return float(np.mean(all_erde) * 100)


@Container.bind()
class ERDE_5(BaseMetric):
    def __init__(self):
        self._erde = ERDE(k=5)

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> float | np.ndarray:
        if y_true is None:
            raise ValueError("y_true must be provided for evaluation")

        return self._erde.evaluate(x_data, y_true, y_pred)


@Container.bind()
class ERDE_50(BaseMetric):
    def __init__(self):
        self._erde = ERDE(k=50)

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> float | np.ndarray:
        if y_true is None:
            raise ValueError("y_true must be provided for evaluation")

        return self._erde.evaluate(x_data, y_true, y_pred)
