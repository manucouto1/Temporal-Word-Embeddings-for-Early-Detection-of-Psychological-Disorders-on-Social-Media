from framework3 import BaseMetric, Container, XYData, Precission

import numpy as np

__all__ = ["PrecissionScore"]


@Container.bind()
class PrecissionScore(BaseMetric):
    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> float | np.ndarray:
        if y_true is None:
            raise ValueError("y_true must be provided for evaluation")

        x_df = x_data.value.copy()
        x_df.loc[:, "label"] = y_true.value
        x_df = x_df.explode(["chunk", "text", "date", "n_texts"]).reset_index(drop=True)
        x_df["_y"] = y_pred.value

        aux = x_df.groupby(["user"]).agg(
            {"label": "first", "_y": lambda x: 1 if any(list(x)) else 0}
        )

        return float(
            Precission(average="binary").evaluate(
                x_data, XYData.mock(aux.label.tolist()), XYData.mock(aux._y.tolist())
            )
        )
