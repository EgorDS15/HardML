import math
import pickle
import random
from typing import List, Tuple

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm.auto import tqdm


class Solution:
    def __init__(self, n_estimators: int = 100, lr: float = 0.5, ndcg_top_k: int = 10,
                 subsample: float = 0.6, colsample_bytree: float = 0.9,
                 max_depth: int = 5, min_samples_leaf: int = 8):
        self._prepare_data()

        self.ndcg_top_k = ndcg_top_k
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        # допишите ваш код здесь

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
        self.ys_train = torch.FloatTensor(y_train).reshape(-1, 1)
        self.ys_test = torch.FloatTensor(y_test).reshape(-1, 1)

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray, inp_query_ids: np.ndarray) -> np.ndarray:
        num_q_and_docs = tuple(zip(inp_query_ids, inp_feat_array))
        set_idx = sorted([i for i in set(inp_query_ids)])
        scaled_data = np.empty((0, 136))

        for idx in set_idx:
            batch = np.asarray([features for group, features in num_q_and_docs if group == idx])
            scaler = StandardScaler()
            scaled_data = np.append(scaled_data, scaler.fit_transform(batch), axis=0)

        return scaled_data

    def _train_one_tree(self, cur_tree_idx: int, train_preds: torch.FloatTensor) -> Tuple[
        DecisionTreeRegressor, np.ndarray]:
        # допишите ваш код здесь
        pass

    def _calc_data_ndcg(self, queries_list: np.ndarray,
                        true_labels: torch.FloatTensor, preds: torch.FloatTensor) -> float:
        # допишите ваш код здесь
        pass

    def fit(self):
        np.random.seed(0)
        # допишите ваш код здесь
        pass

    def predict(self, data: torch.FloatTensor) -> torch.FloatTensor:
        # допишите ваш код здесь
        pass

    def _compute_lambdas(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        # рассчитаем нормировку, IdealDCG
        values, rank_order = torch.sort(y_true, descending=True, axis=0)
        ideal_dcg = float(2 ** values[0] - 1)
        for ind, val in zip(range(2, len(values) + 1), values[1:]):
            dif = float(2 ** val - 1) / math.log2(ind + 1)
            ideal_dcg += dif

        N = 1 / ideal_dcg

        # рассчитаем порядок документов согласно оценкам релевантности и получаем индексы отсортированных элементов
        values, rank_order = torch.sort(y_true, descending=True, axis=0)
        # 2 print(rank_order)
        # Добавляем 1, чтобы позиция начинала отсчет с 1, а не от нуля
        rank_order += 1
        # 3 print(rank_order)

        with torch.no_grad():
            # получаем все попарные разницы скоров в батче. Разность каждого с каждым
            pos_pairs_score_diff = 1.0 + torch.exp((y_pred - y_pred.t()))
            # 4 print(pos_pairs_score_diff)

            # поставим разметку(матрицу) для пар:
            # 1 если первый документ релевантнее
            # -1 если второй документ релевантнее
            # compute_labels_in_batch
            rel_diff = y_true - y_true.t()
            # 1 в этой матрице - объект более релевантен
            pos_pairs = (rel_diff > 0).type(torch.float32)
            # 1 тут - объект менее релевантен
            neg_pairs = (rel_diff < 0).type(torch.float32)
            Sij = pos_pairs - neg_pairs
            # 5 print(Sij)

            # compute_gain_diff
            # посчитаем изменение gain из-за перестановок
            gain_diff = torch.pow(2.0, y_true) - torch.pow(2.0, y_true.t())
            # 6 print(gain_diff)

            # посчитаем изменение знаменателей-дискаунтеров
            decay_diff = (1.0 / torch.log2(rank_order + 1.0)) - (1.0 / torch.log2(rank_order.t() + 1.0))
            # посчитаем непосредственное изменение nDCG
            delta_ndcg = torch.abs(N * gain_diff * decay_diff)
            # посчитаем лямбды
            lambda_update = (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff) * delta_ndcg
            lambda_update = torch.sum(lambda_update, dim=1, keepdim=True)

            return lambda_update

    def _ndcg_k(self, ys_true, ys_pred, ndcg_top_k) -> float:
        # допишите ваш код здесь
        pass

    def save_model(self, path: str):
        pass

    def load_model(self, path: str):
        # допишите ваш код здесь
        pass