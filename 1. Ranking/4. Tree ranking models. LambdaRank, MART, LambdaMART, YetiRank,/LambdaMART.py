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
    def __init__(self, n_estimators: int = 1, lr: float = 0.5, ndcg_top_k: int = 10,
                 subsample: float = 0.6, colsample_bytree: float = 0.9,
                 max_depth: int = 5, min_samples_leaf: int = 8):
        self._prepare_data()

        self.ndcg_top_k = ndcg_top_k
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        # допишите ваш код здесь.
        # Сохраняем обученные деревья
        self.trees = []
        self.train_idx = sorted(list(set(self.query_ids_train)))
        self.test_idx = sorted(list(set(self.query_ids_test)))

        # суммарные предсказания всех предыдущих деревьев(для расчёта лямбд)
        self.tree_preds = torch.zeros(self.X_train.shape[0])

    #         # Инициализируем начальное приближение для каждого объекта тренировочной выборки и сюда
    #         self.lambdas = np.zeros(len(self.tree_preds))

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

        # здесь итерируемся по входящему массиву с запросами, так как это и для трейна и для теста
        for idx in set_idx:
            batch = np.asarray([features for group, features in num_q_and_docs if group == idx])
            scaler = StandardScaler()
            scaled_data = np.append(scaled_data, scaler.fit_transform(batch), axis=0)

        return scaled_data

    def _train_one_tree(self, cur_tree_idx: int,
                        train_preds: torch.FloatTensor) -> Tuple[DecisionTreeRegressor, np.ndarray]:
        """
        Метод для тренировки одного дерева.
        Принимает на вход
            cur_tree_idx — номер текущего дерева, который предлагается использовать в качестве random_seed для того,
                            чтобы алгоритм был детерминирован.
            train_preds — суммарные предсказания всех предыдущих деревьев (для расчёта лямбд).
        В рамках метода необходимо рассчитать лямбды для каждой группы в тренировочном наборе данных,
        затем применить метод случайных подпространств, сделав срез по признакам (случайно выбранная группа, размер
        которой задан параметром colsample_bytree) и по объектам (тоже случайно выбранная группа, размер зависит от
        параметра subsample). Затем произвести тренировку одного DecisionTreeRegressor. Возвращаемые значения — это
        само дерево и индексы признаков, на которых обучалось дерево.
        """

        for idx in self.train_idx[:1]:
            # Маска для выделения определенных мест-значений в одномерном тензоре для обновлений лямбд для своих запросов
            mask = self.query_ids_train == idx
            lambda_batch = self.lambda_arr[mask]
            y_batch = self.ys_train[mask]
            # print(idx, len(y_batch))
            # чтобы оптимизировать лямбду и предсказанные вероятности, нужно идти в сторону антиградиента лямбды
            self.lambda_arr[mask] = self._compute_lambdas(y_batch, lambda_batch)
            print(idx)
            # print(cur_tree_idx, idx, self.lambda_arr, len(self.lambda_arr))
        #             self.lambda_arr = np.append(self.lambda_arr, self.lambdas)

        self.model = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                           random_state=cur_tree_idx)

        # Заменить np.array(0) на индексы подпространств
        return (self.model, np.array(0))

    def fit(self):
        """
         генеральный метод обучения K деревьев, каждое из которых тренируется с использованием метода _train_one_tree.
         Изначальные предсказания до обучения предлагается приравнять к нулю и от этих значений отталкиваться при обучении
         первого дерева. Все обученные деревья необходимо сохранить в список, хранящийся в атрибуте trees класса Solution.
         Для простоты и ускорения работы предлагается рассчитывать предсказания для всех тренировочных и валидационных данных
         после обучения каждого дерева (но досчитывать только изменения за последнее дерево, храня в памяти предсказания
         всех предыдущих).
        """
        np.random.seed(0)
        # Инициализируем начальное приближение нулями. Размер тензора по количеству запросов
        F = torch.zeros(self.X_train.shape[0])

        for tree_ind in range(6):  # self.n_estimators
            # Инициализируем начальное предсказание лямбды нулями. Размер тензора по количеству ответов
            self.lambda_arr = torch.zeros(len(self.ys_train)).reshape(-1, 1)

            # Находим лямбду на каждом дереве для каждого запроса
            tree = self._train_one_tree(tree_ind, self.lambda_arr)
            # Добавляем в список обученную модель
            self.trees.append(tree[0])
            # Если возвращать _train_one_tree должен список с моделью и индексами признакового подпространства
            print(self.lambda_arr)
        print(self.trees)

    def predict(self, data: torch.FloatTensor) -> torch.FloatTensor:
        # допишите ваш код здесь
        self.predictions = []
        pass

    def _compute_lambdas(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        # рассчитаем нормировку, IdealDCG
        values, rank_order = torch.sort(y_true, descending=True, axis=0)
        ideal_dcg = float(2 ** values[0] - 1)
        for ind, val in zip(range(2, len(values) + 1), values[1:]):
            dif = float(2 ** val - 1) / math.log2(ind + 1)
            ideal_dcg += dif

        ###############################################################################################################
        #         Что с этим делать?
        if ideal_dcg > 0:
            N = 1 / ideal_dcg
        else:
            N = 0.01
        ###############################################################################################################
        # рассчитаем порядок документов согласно оценкам релевантности и получаем индексы отсортированных элементов
        values, rank_order = torch.sort(y_true, descending=True, axis=0)
        # Добавляем 1, чтобы позиция начинала отсчет с 1, а не от нуля
        rank_order += 1

        with torch.no_grad():
            # получаем все попарные разницы скоров в батче. Разность каждого с каждым
            pos_pairs_score_diff = 1.0 + torch.exp((y_pred - y_pred.t()))

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

            # compute_gain_diff
            # посчитаем изменение gain из-за перестановок
            gain_diff = torch.pow(2.0, y_true) - torch.pow(2.0, y_true.t())

            # посчитаем изменение знаменателей-дискаунтеров
            decay_diff = (1.0 / torch.log2(rank_order + 1.0)) - (1.0 / torch.log2(rank_order.t() + 1.0))
            # посчитаем непосредственное изменение nDCG
            delta_ndcg = torch.abs(N * gain_diff * decay_diff)
            # посчитаем лямбды
            lambda_update = (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff) * delta_ndcg
            lambda_update = torch.sum(lambda_update, dim=1, keepdim=True)

            return lambda_update  # Возвращать должно torch.FloatTensor

    def _calc_data_ndcg(self, queries_list: np.ndarray, true_labels: torch.FloatTensor,
                        preds: torch.FloatTensor) -> float:
        ndcgs = []

        for it in self.train_idx:
            mask = queries_list == it
            batch_true = self.true_labels[mask]
            batch_pred = self.preds[mask]
            batch_loss = self._ndcg_k(batch_true, batch_pred, self.ndcg_top_k)
            ndcgs.append(batch_loss)

        return np.mean(ndcgs)

    def _ndcg_k(self, ys_true, ys_pred, ndcg_top_k) -> float:
        # допишите ваш код здесь
        if ndcg_top_k > len(ys_pred):
            ndcg_top_k = len(ys_pred)

        sorted_y = sorted(list(zip(ys_true, ys_pred)), key=lambda x: x[1], reverse=True)
        res_list = [x[0].item() for x in sorted_y]

        exp_res = 2 ** res_list[0] - 1
        for ind, val in zip(range(2, ndcg_top_k + 1), res_list[1:ndcg_top_k]):
            dif = (2 ** val - 1) / math.log2(ind + 1)
            exp_res += dif

        sorted_y_ideal = ys_true.sort(descending=True).values
        idcg = 2 ** sorted_y_ideal[0] - 1
        for ind, val in zip(range(2, ndcg_top_k + 1), sorted_y_ideal[1:ndcg_top_k]):
            dif = (2 ** val - 1) / math.log2(ind + 1)
            idcg += dif

        return float(exp_res / idcg)

    def save_model(self, path: str):
        pass

    def load_model(self, path: str):
        # допишите ваш код здесь
        pass