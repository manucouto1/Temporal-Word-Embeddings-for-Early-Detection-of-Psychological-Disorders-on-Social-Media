from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Literal
from scipy.sparse import dok_matrix
from tqdm import tqdm
from framework3.base.base_clases import BaseFilter
from framework3.base.base_types import XYData
from framework3.container import Container

import pandas as pd
import numpy as np
import torch
import os
import time

from src.models.twec import TWEC
from src.models.deltas import DISTANCES


@Container.bind()
class TWECFilter(BaseFilter):
    def __init__(
        self,
        context_size: int,
        _cpus: int = 4,
        deltas_f: List[
            Literal[
                "cosine",
                "euclidean",
                "chebyshev",
                "jensen_shannon",
                "wasserstein",
                "manhattan",
                "minkowski",
            ]
        ] = ["cosine"],
    ):
        super().__init__()
        self._twec = TWEC(size=300, window=context_size)
        self.deltas_f = deltas_f
        self.context_size = context_size
        actual_cpus = os.cpu_count()
        if actual_cpus is not None:
            self._cpus = min(actual_cpus, _cpus)
        else:
            self._cpus = _cpus

    def fit(self, x: XYData, y: XYData | None) -> float | None:
        start = time.time()

        data: pd.DataFrame = x.value
        self._twec.train_compass(data.text.values.tolist())
        self._vocab_hash_map = dict(
            zip(
                self._twec.compass.wv.index_to_key,  # type: ignore
                range(len(self._twec.compass.wv.index_to_key)),  # type: ignore
            )
        )
        end = time.time()
        print(f"* Twec train last: {end - start:.6f} seconds")

    def predict(self, x: XYData) -> XYData:
        start = time.time()
        data: pd.DataFrame = x.value
        n_rows = len(data.index)
        n_cols = len(self._vocab_hash_map.items())
        metric_names = self.deltas_f

        all_deltas = {
            metric: dok_matrix((n_rows, n_cols), dtype=np.float32)
            for metric in metric_names
        }

        def process_user_deltas(i, tc):
            result = {metric: [] for metric in metric_names}
            for word in tc.wv.index_to_key:  # type: ignore
                if word in self._vocab_hash_map:
                    j = self._vocab_hash_map[word]
                    for metric in metric_names:
                        dist = (
                            DISTANCES[metric](
                                torch.tensor(np.array([[self._twec.compass.wv[word]]])),  # type: ignore
                                torch.tensor(np.array([[tc.wv[word]]])),  # type: ignore
                            )
                            .detach()
                            .cpu()
                            .item()
                        )
                        result[metric].append((i, j, dist))
            return result

        with ThreadPoolExecutor(max_workers=self._cpus) as executor:
            futures = {
                executor.submit(
                    process_user_deltas, i, self._twec.train_slice(row.text)
                ): i
                for i, row in tqdm(
                    enumerate(data.itertuples()),
                    total=n_rows,
                    desc="generating embeddings",
                )
            }

            for future in tqdm(
                as_completed(futures), total=n_rows, desc="parallel prediction"
            ):
                chunk_result = future.result()
                for metric, values in chunk_result.items():
                    for i, j, val in values:
                        all_deltas[metric][i, j] = val
        end = time.time()
        print(f"* Twec test last: {end - start:.6f} seconds")
        return XYData.mock(all_deltas)
