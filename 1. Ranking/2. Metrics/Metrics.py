from math import log2
from torch import Tensor, sort


def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int:
    num_correct = 0
    set_pairs = set(sorted(tuple(zip(ys_true.tolist(), ys_pred.tolist())), key=lambda x: x[0], reverse=True))
    for i, j in set_pairs:
        if i == j:
            num_correct += 1
    return len(set_pairs) - num_correct


def compute_gain(y_value: float, gain_scheme: str) -> float:
    if gain_scheme == 'const':
        return y_value
    elif gain_scheme == 'exp2':
        return 2**y_value - 1
    else:
        raise ValueError('Parameter gain_scheme takes only "const" or "exp2" value!')


def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str) -> float:
    sorted_y = sorted(list(zip(ys_true, ys_pred)), key=lambda x: x[1], reverse=True)
    res_list = [x[0].item() for x in sorted_y]
    if gain_scheme == 'const':
        const_res = res_list[0]
        # Создаем кортеж из индекса и значения
        for ind, val in zip(range(2, len(res_list)+1), res_list[1:]):
            dif = val / log2(ind+1)
            const_res += dif
        return const_res
    elif gain_scheme == 'exp2':
        const_res = compute_gain(res_list[0], gain_scheme)
        for ind, val in zip(range(2, len(res_list)+1), res_list[1:]):
            dif = compute_gain(val, gain_scheme) / log2(ind + 1)
            const_res += dif
        return const_res


def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    sorted_y = ys_true.sort(descending=True).values
    if gain_scheme == 'const':
        idcg = sorted_y[0]
        # Создаем кортеж из индекса и значения
        for ind, val in zip(range(2, len(sorted_y)+1), sorted_y[1:]):
            dif = val/log2(ind+1)
            idcg += dif
    elif gain_scheme == 'exp2':
        idcg = compute_gain(sorted_y[0], gain_scheme)
        for ind, val in zip(range(2, len(sorted_y)+1), sorted_y[1:]):
            dif = compute_gain(val, gain_scheme) / log2(ind + 1)
            idcg += dif
    return float(dcg(ys_true, ys_pred, gain_scheme) / idcg)


# def precission_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
#     if ys_true.sum() == 0:
#         return -1
#
#     if k > len(ys_pred):
#         k = len(ys_pred)
#
#     y_labels = []
#     for i in ys_pred[:k]:
#         if (i < 0.5) & (i >= 0.0):
#             y_labels.append(0.0)
#         elif (i >= 0.5) & (i <= 1.0):
#             y_labels.append(1.0)
#
#     tp = 0
#     tn = 0
#     fp = 0
#     fn = 0
#     for i, j in zip(ys_true[:k], y_labels):
#         if (i == 1) & (j == 1):
#             tp += 1
#         elif (i == 1) & (j == 0):
#             fn += 1
#         elif (i == 0) & (j == 1):
#             fp += 1
#         elif (i == 0) & (j == 0):
#             tn += 1
#         else:
#             raise ValueError('Incorrect label in y_true')
#
#     #     recall = tp / (tp + fn)
#     precision_k = tp / (tp + fp)
#     return precision_k

def precission_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    if ys_true.sum() == 0:
        return -1

    if k > len(ys_pred):
        k = len(ys_pred)

    sorted_y = sorted(tuple(zip(ys_true, ys_pred)), key=lambda x: x[1], reverse=True)
    res = 0
    for true_l, proba in sorted_y[:k]:
        res += true_l * proba

    return float(res / k)


def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    sorted_tuples = sorted(tuple(zip(ys_true, ys_pred)), key=lambda x: x[1], reverse=True)

    for rank, tup in enumerate(sorted_tuples, start=1):
        if tup[0] == 1:
            return 1 / rank


def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15 ) -> float:
    # допишите ваш код здесь
    pass


def average_precision(ys_true: Tensor, ys_pred: Tensor) -> float:
    sorted_y = sorted(tuple(zip(ys_true, ys_pred)), key=lambda x: x[1], reverse=True)

    if ys_true.sum() == 0:
        return -1

    test = []
    for num, val in enumerate(sorted_y, start=1):
        test.append((num, val[0], val[1]))

    res = 0
    positive_number = 0
    for y_tuple in test:

        if y_tuple[1] == 1:
            positive_number += 1
            print(y_tuple)
            res += positive_number / y_tuple[0]

    return res / positive_number
