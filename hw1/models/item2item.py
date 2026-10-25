from typing import Iterable

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix


class ItemItemCF:
    def __init__(self, n_neighbors=40, min_common=3, shrinkage=5.0, damping=2.0):
        self.n_neighbors = int(n_neighbors)
        self.min_common = int(min_common)
        self.shrinkage = float(shrinkage)
        self.damping = float(damping)

    def fit(self, votes: pl.DataFrame, catalog_ids: Iterable[int] | None = None):
        self.user_ids = np.sort(votes["user_id"].unique().to_numpy())
        rated_item_ids = np.sort(votes["article_id"].unique().to_numpy())
        if catalog_ids is None:
            catalog_ids = rated_item_ids
        self.catalog_ids = np.sort(np.unique(np.asarray(list(catalog_ids), dtype=np.int64)))
        self.item_ids = rated_item_ids
        self.user_to_idx = {int(value): i for i, value in enumerate(self.user_ids)}
        self.item_to_idx = {int(value): i for i, value in enumerate(self.item_ids)}

        # Popularity is used for cold users
        # counts = votes.group_by("article_id").len().sort(
        #     ["len", "article_id"], descending=[True, False]
        # )
        # self.popular_item_ids = counts["article_id"].to_list()
        # self.popularity = dict(counts.select("article_id", "len").iter_rows())

        MIN_RATINGS = 20
        counts = votes.group_by("article_id").agg(
            pl.len().alias("n_ratings"),
            pl.col("rate").mean().alias("avg_rating"),
        ).filter(
            pl.col("n_ratings") >= MIN_RATINGS
        ).sort(
            ["avg_rating", "n_ratings", "article_id"],
            descending=[True, True, False],
        )
        self.popular_item_ids = counts["article_id"].to_list()
        self.popularity = dict(
            counts.select("article_id", "avg_rating").iter_rows()
        )
        
        user_idx = np.array([self.user_to_idx[int(x)] for x in votes["user_id"]])
        item_idx = np.array([self.item_to_idx[int(x)] for x in votes["article_id"]])
        ratings = votes["rate"].to_numpy().astype(float)

        user_counts = np.bincount(user_idx, minlength=len(self.user_ids))
        user_sums = np.bincount(user_idx, weights=ratings, minlength=len(self.user_ids))
        global_mean = float(np.mean(ratings))
        user_means = np.divide(
            user_sums,
            user_counts,
            out=np.full(len(self.user_ids), global_mean),
            where=user_counts > 0,
        )
        residuals = ratings - user_means[user_idx]

        shape = (len(self.user_ids), len(self.item_ids))
        centered = csr_matrix((residuals, (user_idx, item_idx)), shape=shape)
        observed = csr_matrix(
            (np.ones(len(votes), dtype=np.float32), (user_idx, item_idx)),
            shape=shape,
        )

        gram = (centered.T @ centered).tocsr()
        overlap = (observed.T @ observed).tocsr()
        norms = np.sqrt(np.maximum(gram.diagonal(), 0))
        neighbors = [dict() for _ in self.item_ids]

        for i in range(len(self.item_ids)):
            row = gram.getrow(i)
            overlap_row = overlap.getrow(i)
            dot_by_item = dict(zip(row.indices, row.data))
            overlap_by_item = dict(zip(overlap_row.indices, overlap_row.data))

            candidates = []
            for j, dot in dot_by_item.items():
                common = int(overlap_by_item.get(j, 0))
                
                if i == j or common < self.min_common:
                    continue
                denominator = norms[i] * norms[j]
                if denominator <= 0:
                    continue
                similarity = (dot / denominator) * common / (common + self.shrinkage)
                if similarity > 0:
                    candidates.append((j, float(similarity), common))

            candidates.sort(key=lambda x: x[1], reverse=True)
            for j, similarity, common in candidates[:self.n_neighbors]:
                neighbors[i][j] = (similarity, common)

        self.neighbors = neighbors
        self.user_means = user_means
        self.user_history = [dict() for _ in self.user_ids]
        for u, i, rating in zip(user_idx, item_idx, ratings):
            self.user_history[u][int(i)] = float(rating - user_means[u])

        return self

    def _user_index(self, user_id):
        return self.user_to_idx.get(int(user_id))

    def recommend(self, user_id: int, n=10) -> pl.DataFrame:
        n = max(0, int(n))
        if n == 0:
            return pl.DataFrame(schema={
                "article_id": pl.Int64, "score": pl.Float64, "reason": pl.String
            })

        user_idx = self._user_index(user_id)
        if user_idx is None:
            # Cold user
            seen = set()
            scored = []
        else:
            history = self.user_history[user_idx]
            seen = {int(self.item_ids[i]) for i in history}
            candidate_scores = {}

            for candidate_idx, candidate_id in enumerate(self.item_ids):
                if int(candidate_id) in seen:
                    continue
                weighted_sum = 0.0
                weight_sum = 0.0
                for neighbor_idx, (similarity, _) in self.neighbors[candidate_idx].items():
                    if neighbor_idx in history:
                        weighted_sum += similarity * history[neighbor_idx]
                        weight_sum += abs(similarity)
                if weight_sum > 0:
                    candidate_scores[int(candidate_id)] = float(
                        self.user_means[user_idx] + weighted_sum / (weight_sum + self.damping)
                    )

            scored = sorted(
                candidate_scores.items(), key=lambda x: (-x[1], x[0])
            )

        # Fill rest with popular items
        popularity_order = [int(x) for x in self.popular_item_ids]
        if user_idx is None:
            scored = [(item_id, float(self.popularity[item_id])) for item_id in popularity_order]
        existing = {item_id for item_id, _ in scored}
        scored.extend(
            (item_id, float(self.popularity.get(item_id, 0)))
            for item_id in popularity_order
            if item_id not in existing and item_id not in seen
        )

        results = [
            {"article_id": item_id, "score": score}
            for item_id, score in scored if item_id not in seen
        ][:n]

        return pl.DataFrame(
            results,
            schema={"article_id": pl.Int64, "score": pl.Float64},
        )

    def nearest_items(self, article_id: int, n=10) -> pl.DataFrame:
        article_idx = self.item_to_idx.get(int(article_id))
        rows = []
        for neighbor_idx, (similarity, common) in self.neighbors[article_idx].items():
            neighbor_id = int(self.item_ids[neighbor_idx])
            rows.append({
                "article_id": neighbor_id,
                "similarity": similarity,
                "shared_users": int(common),
            })

        rows.sort(key=lambda row: (-row["similarity"], row["article_id"]))
        return pl.DataFrame(
            rows[:max(0, int(n))],
            schema={
                "article_id": pl.Int64,
                "similarity": pl.Float64,
                "shared_users": pl.Int64,
            },
        )
