import math

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler

from typing import List


class ListNet(torch.nn.Module):
    def __init__(self, num_input_features: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # укажите архитектуру простой модели здесь
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_input_features, self.hidden_dim),
            torch.nn.Softmax(dim=1),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(0.6),
            torch.nn.Linear(self.hidden_dim, 15),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(15, 1)
        )

    def forward(self, input_1: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_1)
        return logits


class Solution:
    def __init__(self, n_epochs: int = 5, listnet_hidden_dim: int = 30, lr: float = 0.001, ndcg_top_k: int = 10):
        self._prepare_data()
        self.num_input_features = self.X_train.shape[1]
        self.ndcg_top_k = ndcg_top_k
        self.n_epochs = n_epochs

        self.model = self._create_model(self.num_input_features, listnet_hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _get_data(self) -> List[np.ndarray]:
        train_df, test_df = msrank_10k()

        X_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)

        X_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)

        return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]

    def _prepare_data(self) -> None:
        (X_train, y_train, self.query_ids_train, X_test, y_test, self.query_ids_test) = self._get_data()
        self.X_train = torch.FloatTensor(self._scale_features_in_query_groups(X_train, self.query_ids_train))
        self.X_test = torch.FloatTensor(self._scale_features_in_query_groups(X_test, self.query_ids_test))
        self.ys_train = torch.FloatTensor(y_train)
        self.ys_test = torch.FloatTensor(y_test)

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray, inp_query_ids: np.ndarray) -> np.ndarray:
        num_q_and_docs = tuple(zip(inp_query_ids, inp_feat_array))
        set_idx = sorted([i for i in set(inp_query_ids)])
        scaled_data = np.empty((0, 136))

        for idx in set_idx:
            batch = np.asarray([features for group, features in num_q_and_docs if group == idx])
            scaler = StandardScaler()
            scaled_data = np.append(scaled_data, scaler.fit_transform(batch), axis=0)

        return scaled_data

    def _create_model(self, listnet_num_input_features: int, listnet_hidden_dim: int) -> torch.nn.Module:
        torch.manual_seed(0)
        net = ListNet(listnet_num_input_features, listnet_hidden_dim)
        return net

    def fit(self) -> List[float]:
        # допишите ваш код здесь
        val_ndcg = []
        for t in range(self.n_epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            self._train_one_epoch()
            val_ndcg.append(self._eval_test_set())
        return val_ndcg

    def _calc_loss(self, batch_ys: torch.FloatTensor, batch_pred: torch.FloatTensor) -> torch.FloatTensor:
        P_y_i = torch.softmax(batch_ys, dim=0)
        P_z_i = torch.softmax(batch_pred, dim=0)

        return -torch.sum(P_y_i * torch.log(P_z_i))  # -torch.sum(P_y_i * torch.log(P_z_i / P_y_i))

    def _train_one_epoch(self) -> None:
        self.model.train()
        ziped_train = list(zip(self.query_ids_train, self.X_train))
        ziped_y_train = list(zip(self.query_ids_train, self.ys_train))
        set_idx = sorted([i for i in set(self.query_ids_train)])

        ids_batch = 0
        for it in set_idx:
            batch_size = len([features for group, features in ziped_train if group == it])
            batch = self.X_train[ids_batch: ids_batch + batch_size]
            batch_y = self.ys_train[ids_batch: ids_batch + batch_size]
            ids_batch += batch_size

            self.optimizer.zero_grad()
            if len(batch) > 0:
                batch_pred = self.model(batch)
                batch_loss = self._calc_loss(batch_y, batch_pred)
                batch_loss.backward()
                self.optimizer.step()

    def _eval_test_set(self) -> float:
        with torch.no_grad():
            self.model.eval()
            ndcgs = []
            ziped_test = list(zip(self.query_ids_test, self.X_test))
            ziped_y_test = list(zip(self.query_ids_test, self.ys_test))
            set_idx = sorted([i for i in set(self.query_ids_test)])

            ids_batch = 0
            for it in set_idx:
                batch_size = len([features for group, features in ziped_test if group == it])
                batch = self.X_test[ids_batch: ids_batch + batch_size]
                batch_y = self.ys_test[ids_batch: ids_batch + batch_size]
                ids_batch += batch_size

                if len(batch) > 0:
                    batch_pred = self.model(batch)
                    # Если NDCG рассчитать невозможно или по каким-то причинам появляется ошибка,
                    # то NDCG=0 (а не пропускается)??????????
                    batch_loss = self._ndcg_k(batch_y, batch_pred, self.ndcg_top_k)
                    ndcgs.append(batch_loss)

            return np.mean(ndcgs)

    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor, ndcg_top_k: int) -> float:
        if ndcg_top_k > len(ys_pred):
            ndcg_top_k = len(ys_pred)

        sorted_y = sorted(list(zip(ys_true, ys_pred)), key=lambda x: x[1], reverse=True)
        res_list = [x[0].item() for x in sorted_y]

        exp_res = 2 ** res_list[0] - 1
        for ind, val in zip(range(2, ndcg_top_k + 1), res_list[1:ndcg_top_k]):
            dif = (2 ** val - 1) / log2(ind + 1)
            exp_res += dif

        sorted_y_ideal = ys_true.sort(descending=True).values
        idcg = 2 ** sorted_y_ideal[0] - 1
        for ind, val in zip(range(2, ndcg_top_k + 1), sorted_y_ideal[1:ndcg_top_k]):
            dif = (2 ** val - 1) / log2(ind + 1)
            idcg += dif

        return float(exp_res / idcg)