from math import log2
from torch import Tensor, sort


def num_swapped_pairs(ys_true: torch.Tensor, ys_pred: torch.Tensor) -> int:
    # Сортируем предсказанные значения и получаем отсортированный массив значений и массив индексов
    # отсортированных значений
    ys_pred_sorted, argsort = torch.sort(ys_pred, descending=True, dim=0)
    # Применяем индексы отсортированных предсказанных значений к истинным значениям и получаем список
    # истинных значений отсортированный по убыванию вероятности
    ys_true_sorted = ys_true[argsort]

    # количество документов
    num_objects = ys_true_sorted.shape[0]
    # стартовый счетчик
    swapped_cnt = 0
    # Берем каждый объект кроме последнего
    for cur_obj in range(num_objects - 1):
        # Теперь начинаем сравнивать текущие значение по индексу ИСТИННЫХ ЗНАЧЕНИЙ со следующим истинным значением
        for next_obj in range(cur_obj + 1, num_objects):
            # Если текущее значение отсортированных истинных значений меньше последующего(то есть ранг
            # вышестоящего документа ниже ранга последующего документа)...
            if ys_true_sorted[cur_obj] < ys_true_sorted[next_obj]:
                # а так же предсказанное значение вероятности(например если предсказывается она) текущего документа
                # выше чем предсказанное значение вероятности следующего(а значит нижестоящего) документа
                if ys_pred_sorted[cur_obj] > ys_pred_sorted[next_obj]:
                    # то считаем, что эта пара упорядочена неверно
                    swapped_cnt += 1
            # А так же если ранг текущего выше следующего
            elif ys_true_sorted[cur_obj] > ys_true_sorted[next_obj]:
                # но вероятность ниже
                if ys_pred_sorted[cur_obj] < ys_pred_sorted[next_obj]:
                    # то так же считаем, что эта пара упорядочена неверно
                    swapped_cnt += 1
    return swapped_cnt

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


# Не принятое решение
def precission_at_k(ys_true: torch.Tensor, ys_pred: torch.Tensor, k: int) -> float:
    if ys_true.sum() == 0:
        return -1
    _, argsort = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[argsort]
    hits = ys_true_sorted[:k].sum()
    prec = hits / min(ys_true.sum(), k)
    return float(prec)


def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    sorted_tuples = sorted(tuple(zip(ys_true, ys_pred)), key=lambda x: x[1], reverse=True)

    for rank, tup in enumerate(sorted_tuples, start=1):
        if tup[0] == 1:
            return 1 / rank


def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15 ) -> float:
    sorted_y = sorted(tuple(zip(ys_true, ys_pred)), key=lambda x: x[1], reverse=True)

    full_tuple = [(i, *j) for i, j in enumerate(sorted_y)]
    full_tuple = [(ind, true) for ind, true, _ in full_tuple]
    p_look = [1]
    p_rel = [full_tuple[0][1]]

    for ind, true_l in full_tuple[1:]:
        print(ind, true_l, p_look, p_rel)
        p_look.append(p_look[ind - 1] * (1 - p_rel[ind - 1]) * (1 - p_break))
        p_rel.append(true_l)

    pfound = 0
    for look, rel in tuple(zip(p_look, p_rel)):
        pfound += look * rel

    return pfound


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
