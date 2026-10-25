from typing import Iterable

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix


class ALS:
    def __init__(
        self,
        factors=64,
        regularization=0.1,
        iterations=20,
        seed=55555,
    ):
        self.factors = int(factors)
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.seed = int(seed)

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

        self.global_mean = float(np.mean(ratings))
        residuals = ratings - self.global_mean

        shape = (len(self.user_ids), len(self.item_ids))
        R = csr_matrix(
            (residuals, (user_idx, item_idx)),
            shape=shape,
            dtype=np.float64,
        )
        R_csc = R.tocsc()

        n_users, n_items = shape
        rng = np.random.default_rng(self.seed)

        # Small random initialization.
        self.user_factors = rng.normal(
            0.0, 0.01, (n_users, self.factors)
        )
        self.item_factors = rng.normal(
            0.0, 0.01, (n_items, self.factors)
        )

        eye = np.eye(self.factors)

        # main ALS
        for _ in range(self.iterations):
            # user step
            for u in range(n_users):
                start, end = R.indptr[u], R.indptr[u + 1]
                items = R.indices[start:end]
                values = R.data[start:end]

                if len(items) == 0:
                    continue

                Q = self.item_factors[items]
                A = Q.T @ Q + self.regularization * eye
                b = Q.T @ values

                self.user_factors[u] = np.linalg.solve(A, b)

            # item step
            for i in range(n_items):
                start, end = R_csc.indptr[i], R_csc.indptr[i + 1]
                users = R_csc.indices[start:end]
                values = R_csc.data[start:end]

                if len(users) == 0:
                    continue

                P = self.user_factors[users]
                A = P.T @ P + self.regularization * eye
                b = P.T @ values

                self.item_factors[i] = np.linalg.solve(A, b)

        self.user_history = [set() for _ in self.user_ids]
        for u, i in zip(user_idx, item_idx):
            self.user_history[u].add(int(i))

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

        scores = (
            self.global_mean
            + self.item_factors @ self.user_factors[user_idx]
        )

        seen = self.user_history[user_idx]
        ranked_indices = np.argsort(-scores, kind="stable")

        results = []
        for item_idx in ranked_indices:
            if item_idx in seen:
                continue

            results.append({
                "article_id": int(self.item_ids[item_idx]),
                "score": float(scores[item_idx]),
            })

            if len(results) >= n:
                break

        return pl.DataFrame(results, schema=schema)

    def predict(self, user_id: int, article_id: int) -> float:
        user_idx = self._user_index(user_id)
        item_idx = self.item_to_idx.get(int(article_id))

        if user_idx is None or item_idx is None:
            return float("nan")

        return float(
            self.global_mean
            + self.user_factors[user_idx]
            @ self.item_factors[item_idx]
        )
