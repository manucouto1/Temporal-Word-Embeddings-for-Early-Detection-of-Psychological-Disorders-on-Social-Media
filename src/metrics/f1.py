from framework3 import BaseMetric, Container, XYData, F1

import numpy as np

__all__ = ["F1Score"]


@Container.bind()
class F1Score(BaseMetric):
    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> float | np.ndarray:
        if y_true is None:
            raise ValueError("y_true must be provided for evaluation")
        print(x_data)
        x_df = x_data.value.copy()
        x_df.loc[:, "label"] = y_true.value
        x_df = x_df.explode(["chunk", "text", "date", "n_texts"]).reset_index(drop=True)
        x_df["_y"] = y_pred.value
        print(x_df)
        aux = x_df.groupby(["user"]).agg(
            {"label": "first", "_y": lambda x: 1 if any(list(x)) else 0}
        )

        return float(
            F1(average="binary").evaluate(
                x_data, XYData.mock(aux.label.tolist()), XYData.mock(aux._y.tolist())
            )
        )
