import pandas as pd

from src.metrics.f1 import F1Score
from src.metrics.precission import PrecissionScore
from src.metrics.recall import RecallScore
from src.filters.delta_selector import DeltaSelectorFilter
from src.filters.svm import ClassifierSVM
from src.filters.twec_filter import TWECFilter
from src.metrics.erde import ERDE_5, ERDE_50

from framework3.base.base_types import XYData
from framework3 import F3Pipeline, Cached, SklearnOptimizer
from rich import print

depression_2017 = (
    "depression 2017",
    pd.read_csv("data/standard_depression_train_2017.csv", index_col=0),
    pd.read_csv("data/standard_depression_test_2017.csv", index_col=0),
)
depression_2018 = (
    "depression 2018",
    pd.read_csv("data/standard_depression_train_2018.csv", index_col=0),
    pd.read_csv("data/standard_depression_2018.csv", index_col=0),
)
depression_2022 = (
    "depression 2022",
    pd.read_csv("data/standard_depression_train_2022.csv", index_col=0),
    pd.read_csv("data/standard_depression_2022.csv", index_col=0),
)

anorexia_2018 = (
    "anorexia 2018",
    pd.read_csv("data/standard_anorexia_train_2018.csv", index_col=0),
    pd.read_csv("data/standard_anorexia_test_2018.csv", index_col=0),
)
anorexia_2019 = (
    "anorexia 2019",
    pd.read_csv("data/standard_anorexia_train_2019.csv", index_col=0),
    pd.read_csv("data/standard_anorexia_2019.csv", index_col=0),
)

galbling_2022 = (
    "galbling 2022",
    pd.read_csv("data/standard_gambling_2021.csv", index_col=0),
    pd.read_csv("data/standard_gambling_2022.csv", index_col=0),
)
galbling_2023 = (
    "galbling 2023",
    pd.read_csv("data/standard_gambling_train_2023.csv", index_col=0),
    pd.read_csv("data/standard_gambling_2023.csv", index_col=0),
)

self_harm_2020 = (
    "self_harm 2020",
    pd.read_csv("data/standard_self-harm_2019.csv", index_col=0),
    pd.read_csv("data/standard_self-harm_2020.csv", index_col=0),
)
self_harm_2021 = (
    "self_harm 2021",
    pd.read_csv("data/standard_self-harm_train_2021.csv", index_col=0),
    pd.read_csv("data/standard_self-harm_2021.csv", index_col=0),
)


datasets = [
    depression_2017,
    depression_2018,
    depression_2022,
    anorexia_2018,
    anorexia_2019,
    galbling_2022,
    galbling_2023,
    self_harm_2020,
    self_harm_2021,
]


def prepare_dataset(name, df, sufix):
    # Data preparation and preprocessing steps...
    # Grouping by user and by chunk
    prep_df = (
        df.groupby(["user", "chunk"])
        .agg(
            {
                "id": "count",
                "text": list,
                "date": list,
                "label": "first",
            }
        )
        .rename(columns={"id": "n_texts"})
        .reset_index()
    )
    x = XYData(_hash=f"{name} {sufix}", _path="/deltas research", _value=prep_df)
    y = XYData(
        _hash=f"{name} {sufix} y",
        _path="/deltas research",
        _value=prep_df.label.tolist(),
    )

    return x, y


def main():
    results = {}
    for name, train, test in datasets:
        print(f"Processing {name}")
        train_x, train_y = prepare_dataset(name, train, "train")
        test_x, test_y = prepare_dataset(name, test, "test")

        # Pipeline configuration
        pipeline = F3Pipeline(
            filters=[
                Cached(
                    filter=TWECFilter(
                        context_size=25,
                        _cpus=10,
                        deltas_f=["cosine", "euclidean", "manhattan", "chebyshev"],
                    ),
                ),
                DeltaSelectorFilter(deltas_f="cosine"),
                F3Pipeline(
                    filters=[
                        ClassifierSVM(
                            tol=0.003,
                            probability=False,
                            decision_function_shape="ovr",
                            kernel="rbf",
                            gamma="scale",
                        ).grid(
                            {
                                "C": [1, 3, 5],
                                "class_weight_1": [{1: 1.5}, {1: 2.5}, {1: 3.0}],
                            }
                        )
                    ],
                ).optimizer(SklearnOptimizer(scoring="f1_weighted", cv=2, n_jobs=-1)),
            ],
            metrics=[
                PrecissionScore(),
                RecallScore(),
                F1Score(),
                ERDE_5(),
                ERDE_50(),
            ],
        )

        pipeline.fit(train_x, train_y)
        _y = pipeline.predict(test_x)

        result = pipeline.evaluate(test_x, test_y, _y)
        results[name] = result
        print(f"{name}: {result}")

    print(results)


if __name__ == "__main__":
    main()
