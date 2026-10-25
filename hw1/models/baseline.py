from typing import Iterable

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix


class PopularityModel:
    def fit(self, votes: pl.DataFrame):
        counts = (
            votes.group_by("article_id")
            .len()
            .sort(["len", "article_id"], descending=[True, False])
        )
        self.popular_item_ids = counts["article_id"].to_list()
        self.popularity = dict(counts.select("article_id", "len").iter_rows())

        self.user_seen = {
            int(user_id): set(items)
            for user_id, items in votes.group_by("user_id").agg(
                pl.col("article_id").unique()
            ).iter_rows()
        }
        return self

    def recommend(self, user_id: int, n=10) -> pl.DataFrame:
        n = max(0, int(n))
        seen = self.user_seen.get(int(user_id), set())

        results = []
        for item_id in self.popular_item_ids:
            if len(results) >= n:
                break
            if item_id in seen:
                continue
            results.append(
                {"article_id": int(item_id), "score": float(self.popularity[item_id])}
            )

        return pl.DataFrame(
            results,
            schema={"article_id": pl.Int64, "score": pl.Float64},
        )
