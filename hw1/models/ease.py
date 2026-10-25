from typing import Iterable

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix


class EASE:
    def __init__(self, regularization=500.0):
        self.regularization = float(regularization)

    def fit(
        self,
        votes: pl.DataFrame,
        catalog_ids: Iterable[int] | None = None,
    ):
        self.user_ids = np.sort(votes["user_id"].unique().to_numpy())
        rated_item_ids = np.sort(votes["article_id"].unique().to_numpy())
        if catalog_ids is None:
            catalog_ids = rated_item_ids
        self.catalog_ids = np.sort(np.unique(np.asarray(list(catalog_ids), dtype=np.int64)))
        self.item_ids = rated_item_ids
        self.user_to_idx = {int(value): i for i, value in enumerate(self.user_ids)}
        self.item_to_idx = {int(value): i for i, value in enumerate(self.item_ids)}

        user_idx = np.array([self.user_to_idx[int(x)] for x in votes["user_id"]])
        item_idx = np.array([self.item_to_idx[int(x)] for x in votes["article_id"]])
        ratings = votes["rate"].to_numpy().astype(float)

        user_counts = np.bincount(user_idx, minlength=len(self.user_ids))
        user_sums = np.bincount(user_idx, weights=ratings, minlength=len(self.user_ids))
        global_mean = float(np.mean(ratings))
        self.user_means = np.divide(
            user_sums,
            user_counts,
            out=np.full(len(self.user_ids), global_mean),
            where=user_counts > 0,
        )
        residuals = ratings - self.user_means[user_idx]

        shape = (len(self.user_ids), len(self.item_ids))
        X = csr_matrix(
            (residuals, (user_idx, item_idx)),
            shape=shape,
            dtype=np.float64,
        )

        gram = (X.T @ X).toarray()
        gram.flat[::len(self.item_ids) + 1] += self.regularization

        P = np.linalg.inv(gram)

        B = P / (-np.diag(P))[None, :]
        np.fill_diagonal(B, 0.0)

        self.B = B
        self.user_history = [dict() for _ in self.user_ids]

        for u, i, rating in zip(user_idx, item_idx, ratings):
            self.user_history[u][int(i)] = float(
                rating - self.user_means[u]
            )

        return self

    def _user_index(self, user_id):
        return self.user_to_idx.get(int(user_id))

    def recommend(self, user_id: int, n=10) -> pl.DataFrame:
        n = max(0, int(n))

        schema = {
            "article_id": pl.Int64,
            "score": pl.Float64,
        }
        if n == 0:
            return pl.DataFrame(schema=schema)

        user_idx = self._user_index(user_id)

        if user_idx is None:
            return pl.DataFrame(schema=schema)

        history = self.user_history[user_idx]
        seen = {
            int(self.item_ids[i])
            for i in history
        }

        user_vector = np.zeros(
            len(self.item_ids),
            dtype=np.float64,
        )
        for item_idx, residual in history.items():
            user_vector[item_idx] = residual

        scores = user_vector @ self.B

        ranked_indices = np.argsort(-scores, kind="stable")

        results = []
        for item_idx in ranked_indices:
            item_id = int(self.item_ids[item_idx])

            if item_id in seen:
                continue

            results.append({
                "article_id": item_id,
                "score": float(
                    self.user_means[user_idx] + scores[item_idx]
                ),
            })

            if len(results) >= n:
                break

        return pl.DataFrame(results, schema=schema)

    def predict(self, user_id: int, article_id: int) -> float:
        user_idx = self._user_index(user_id)
        item_idx = self.item_to_idx.get(int(article_id))

        if user_idx is None or item_idx is None:
            return float("nan")

        history = self.user_history[user_idx]
        user_vector = np.zeros(
            len(self.item_ids),
            dtype=np.float64,
        )

        for idx, residual in history.items():
            user_vector[idx] = residual

        return float(
            self.user_means[user_idx]
            + (user_vector @ self.B)[item_idx]
        )
