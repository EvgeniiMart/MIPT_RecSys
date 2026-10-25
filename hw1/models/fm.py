
from typing import Iterable

import numpy as np
import polars as pl
import torch
from scipy.sparse import csr_matrix


class FM:
    def __init__(
        self,
        factors=32,
        learning_rate=0.01,
        regularization=0.0001,
        epochs=20,
        batch_size=4096,
        use_authors=True,
        seed=55555,
    ):
        self.factors = int(factors)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.use_authors = bool(use_authors)
        self.seed = int(seed)

    def fit(
        self,
        votes: pl.DataFrame,
        article_tags: pl.DataFrame,
        article_authors: pl.DataFrame,
        catalog_ids: Iterable[int] | None = None,
    ):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        self.user_ids = np.sort(votes["user_id"].unique().to_numpy())
        self.item_ids = np.sort(votes["article_id"].unique().to_numpy())

        if catalog_ids is None:
            catalog_ids = self.item_ids

        self.catalog_ids = np.sort(
            np.unique(np.asarray(list(catalog_ids), dtype=np.int64))
        )

        self.user_to_idx = {
            int(x): i for i, x in enumerate(self.user_ids)
        }
        self.item_to_idx = {
            int(x): i for i, x in enumerate(self.item_ids)
        }

        # Build metadata
        tag_pairs = (
            article_tags
            .select("article_id", "tag_id")
            .unique()
            .iter_rows()
        )
        self.tags_by_item = {}
        for article_id, tag_id in tag_pairs:
            self.tags_by_item.setdefault(int(article_id), set()).add(
                int(tag_id)
            )

        self.authors_by_item = {}
        if self.use_authors:
            author_pairs = (
                article_authors
                .select("article_id", "user_id")
                .unique()
                .iter_rows()
            )
            for article_id, author_id in author_pairs:
                self.authors_by_item.setdefault(
                    int(article_id), set()
                ).add(int(author_id))

        tag_ids = sorted({
            tag
            for tags in self.tags_by_item.values()
            for tag in tags
        })
        self.tag_to_idx = {
            tag: len(self.user_ids) + len(self.item_ids) + j
            for j, tag in enumerate(tag_ids)
        }

        offset = len(self.user_ids) + len(self.item_ids) + len(tag_ids)
        self.author_to_idx = {}

        if self.use_authors:
            author_ids = sorted({
                author
                for authors in self.authors_by_item.values()
                for author in authors
            })
            self.author_to_idx = {
                author: offset + j
                for j, author in enumerate(author_ids)
            }
            offset += len(author_ids)

        self.user_offset = 0
        self.item_offset = len(self.user_ids)
        self.n_features = offset

        self.global_mean = float(votes["rate"].mean())
        ratings = votes["rate"].to_numpy().astype(np.float32)

        user_indices = np.array([
            self.user_to_idx[int(x)] for x in votes["user_id"]
        ])
        item_indices = np.array([
            self.item_to_idx[int(x)] for x in votes["article_id"]
        ])

        # feature matrix
        X = self._build_matrix(user_indices, item_indices)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.linear = torch.nn.Embedding(self.n_features, 1).to(device)
        self.latent = torch.nn.Embedding(
            self.n_features, self.factors
        ).to(device)

        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.normal_(self.latent.weight, std=0.01)

        self.bias = torch.nn.Parameter(
            torch.tensor(self.global_mean, dtype=torch.float32, device=device)
        )

        optimizer = torch.optim.Adam(
            [
                {"params": self.linear.parameters()},
                {"params": self.latent.parameters()},
                {"params": [self.bias]},
            ],
            lr=self.learning_rate,
        )

        n = len(ratings)

        for _ in range(self.epochs):
            order = rng.permutation(n)

            for start in range(0, n, self.batch_size):
                batch_idx = order[start:start + self.batch_size]
                X_batch = self._to_torch_sparse(X[batch_idx])
                y = torch.as_tensor(
                    ratings[batch_idx], device=device
                )

                # First-order X @ w
                linear_term = torch.sparse.mm(
                    X_batch, self.linear.weight
                ).squeeze(1)

                # Second-order
                summed = torch.sparse.mm(
                    X_batch, self.latent.weight
                )
                squared_sum = summed.square()

                squared_embeddings = self.latent.weight.square()
                summed_squares = torch.sparse.mm(
                    X_batch, squared_embeddings
                )

                interaction = 0.5 * (
                    squared_sum - summed_squares
                ).sum(dim=1)

                predictions = self.bias + linear_term + interaction
                loss = torch.mean((predictions - y) ** 2)

                # L2 penalty
                row_indices = X_batch.indices()[1]
                unique_features = torch.unique(row_indices)

                if unique_features.numel() > 0:
                    loss = loss + self.regularization * (
                        self.linear.weight[unique_features].square().sum()
                        + self.latent.weight[unique_features].square().sum()
                    ) / len(batch_idx)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.user_history = [set() for _ in self.user_ids]
        for u, i in zip(user_indices, item_indices):
            self.user_history[u].add(int(i))

        return self

    def _build_matrix(self, user_indices, item_indices):
        rows, cols, data = [], [], []

        for row, (u, i) in enumerate(zip(user_indices, item_indices)):
            rows.extend([row, row])
            cols.extend([
                self.user_offset + int(u),
                self.item_offset + int(i),
            ])
            data.extend([1.0, 1.0])

            article_id = int(self.item_ids[i])

            for tag in self.tags_by_item.get(article_id, ()):
                feature = self.tag_to_idx.get(tag)
                if feature is not None:
                    rows.append(row)
                    cols.append(feature)
                    data.append(1.0)

            if self.use_authors:
                for author in self.authors_by_item.get(article_id, ()):
                    feature = self.author_to_idx.get(author)
                    if feature is not None:
                        rows.append(row)
                        cols.append(feature)
                        data.append(1.0)

        return csr_matrix(
            (np.asarray(data, dtype=np.float32), (rows, cols)),
            shape=(len(user_indices), self.n_features),
            dtype=np.float32,
        )

    def _to_torch_sparse(self, matrix):
        coo = matrix.tocoo()
        indices = torch.as_tensor(
            np.vstack([coo.row, coo.col]),
            dtype=torch.long,
            device=self.device,
        )
        values = torch.as_tensor(
            coo.data, dtype=torch.float32, device=self.device
        )
        return torch.sparse_coo_tensor(
            indices,
            values,
            size=coo.shape,
            device=self.device,
        ).coalesce()

    def _user_index(self, user_id):
        return self.user_to_idx.get(int(user_id))

    def _score_pairs(self, user_id, article_ids):
        user_idx = self._user_index(user_id)
        if user_idx is None:
            return np.full(len(article_ids), np.nan)

        item_indices = [
            self.item_to_idx.get(int(x), -1) for x in article_ids
        ]
        valid = np.array([i >= 0 for i in item_indices])
        scores = np.full(len(article_ids), np.nan, dtype=np.float64)

        if not valid.any():
            return scores

        positions = np.flatnonzero(valid)
        valid_items = np.asarray(item_indices)[valid]
        X = self._build_matrix(
            np.full(len(valid_items), user_idx),
            valid_items,
        )
        X_torch = self._to_torch_sparse(X)

        with torch.no_grad():
            linear_term = torch.sparse.mm(
                X_torch, self.linear.weight
            ).squeeze(1)

            summed = torch.sparse.mm(X_torch, self.latent.weight)
            summed_squares = torch.sparse.mm(
                X_torch, self.latent.weight.square()
            )

            interaction = 0.5 * (
                summed.square() - summed_squares
            ).sum(dim=1)

            prediction = self.bias + linear_term + interaction

        scores[positions] = prediction.cpu().numpy()
        return scores

    def predict(self, user_id: int, article_id: int) -> float:
        return float(self._score_pairs(user_id, [article_id])[0])

    def recommend(self, user_id: int, n=10) -> pl.DataFrame:
        schema = {
            "article_id": pl.Int64,
            "score": pl.Float64,
        }
        n = max(0, int(n))
        if n == 0:
            return pl.DataFrame(schema=schema)

        user_idx = self._user_index(user_id)
        if user_idx is None:
            return pl.DataFrame(schema=schema)

        seen = self.user_history[user_idx]
        candidate_ids = np.asarray([
            int(item_id)
            for item_id in self.item_ids
            if int(item_id) not in seen
        ], dtype=np.int64)

        if len(candidate_ids) == 0:
            return pl.DataFrame(schema=schema)

        scores = self._score_pairs(user_id, candidate_ids)
        valid = np.isfinite(scores)
        candidate_ids = candidate_ids[valid]
        scores = scores[valid]

        order = np.argsort(-scores, kind="stable")[:n]

        return pl.DataFrame({
            "article_id": candidate_ids[order],
            "score": scores[order],
        }, schema=schema)
