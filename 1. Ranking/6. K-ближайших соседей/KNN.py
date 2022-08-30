from collections import OrderedDict, defaultdict
from typing import Callable, Tuple, Dict, List

import numpy as np
from tqdm.auto import tqdm


def distance(pointA: np.ndarray, documents: np.ndarray) -> np.ndarray:
    res = np.empty((0, len(pointA)))
    for i in documents:
        dif = (pointA - i) ** 2
        sum_difs = np.sum(dif)
        sqre_sum_dif = np.sqrt(sum_difs)
        res = np.append(res, sqre_sum_dif)

    return res.reshape(-1, 1)   # или просто np.linalg.norm(pointA - documents, axis=1, keepdims-True)


def create_sw_graph(
        data: np.ndarray,
        num_candidates_for_choice_long: int = 10, num_edges_long: int = 5,
        num_candidates_for_choice_short: int = 10, num_edges_short: int = 5,
        use_sampling: bool = False, sampling_share: float = 0.05,
        dist_f: Callable = distance) -> Dict[int, List[int]]:
    # Графы будут хранится в словаре-индекс вершины и индексы ее соседей, с которыми она соединена ребрами, тогда
    # становится удобно проверять, что для одной точки другая является соседом. Используем defaultdict так как у нас
    # несколько соседних точек - list. То есть для каждого значение по умолчанию это список, в который мы можем
    # добавлять значения через .append не проверяя наличие ключа(он будет создаваться автоматически при его отсутствии)
    edges = defaultdict(list)
    # Будем итерироваться по всем точкам будущего графа и для каждой точки находить соседей(ближайших и дальних)
    num_points = data.shape[0]

    # За объект берем индекс объекта
    for cur_point_idx in tqdm(range(num_points)):
        # без семплирования
        if not use_sampling:
            # рассчитываем дистанцию между первым объектом и всеми остальными в датасете. Соостветственно самое
            # короткое расстояние будет до самого себя, будет равно нулю
            all_dists = dist_f(data[cur_point_idx, :], data)
            # Сортируем все расстояния и берем начиная с первого, так как на первом месте будет сам объект, так как
            # расстояние между самими собой равно нулю
            argsorted = np.argsort(all_dists.reshape(1, -1))[0][1:]
        # семплинг в этом случае подразумевает выбор подмножества для отбора кондидатов
        else:
            # Определяем долю для конечной подвыборки
            sample_size = int(num_points * sampling_share)
            # Выьираем случайным образом индексы объектов этой подвыборки
            choiced = np.random.choice(list(range(num_points)), size=sample_size, replace=False)
            # расчитваем расстояния от текущего до объектов подвыборки
            part_dists = dist_f(data[cur_point_idx, :], data[choiced, :])
            # сортируем
            argsorted = choiced[np.argsort(part_dists.reshape(1, -1))[0][1:]]
        # Случайный отбор производим, чтобы не брать все объекты из близко расположенных друг к другу объектов.
        # То есть если отбираем самых дальних и есть в дали облако объектов, которые находятся близко друг к другу,
        # то нет смысла брать их все, достаточно будет несколько(одного). Таким образом мы охватим таких кластеров
        # гораздо больше, а не один или два, тем самым сохранив гораздо больше путей
        # выбираем N самых близких кандидатов, из которых мы будем выбирать соседей для точки
        short_cands = argsorted[:num_candidates_for_choice_short]
        # выбираем случайным образом n соседей. Они и будут кондидатами на запись в словарь
        short_choice = np.random.choice(short_cands, size=num_edges_short, replace=False)

        # так же отбираем наиболее дальних объектов
        long_cands = argsorted[-num_candidates_for_choice_long:]
        long_choice = np.random.choice(long_cands, size=num_edges_long, replace=False)
        # конкатенируем и добавляем в граф под индексом объекта. индекс: список ближайших соседей и дальних
        for i in np.concatenate([short_choice, long_choice]):
            edges[cur_point_idx].append(i)

    return dict(edges)


def calc_d_and_upd(all_visited_points: OrderedDict, query_point: np.ndarray,
                   all_documents: np.ndarray, point_idx: int, dist_f: Callable
                   ) -> Tuple[float, bool]:
    # если индекс объекта находится в графе, то есть уже посчтитано расстояние
    if point_idx in all_visited_points:
        # то возвращаем имеющееся расстояние и булево значение как факт того, что рассчет уже был
        return all_visited_points[point_idx], True
    # в противном случае происходить расчет расстояния
    cur_dist = dist_f(query_point, all_documents[point_idx, :].reshape(1, -1))[0][0]
    # и записываем в граф индекс: расстояние
    all_visited_points[point_idx] = cur_dist
    return cur_dist, False


def nsw(query_point: np.ndarray,
        all_documents: np.ndarray,
        graph_edges: Dict[int, List[int]],
        search_k: int = 10,
        num_start_points: int = 5,
        dist_f: Callable = distance) -> np.ndarray:
    # Есть количество точек которые мы пройдем-проинициализируем, и есть количество точек, которые мы должны вернуть
    # Можно из 20 найти 5 и быть довольным, у можно пройти 100 и качество улучшится, так как у нас будет
    # больше информации о расстояниях
    all_visited_points = OrderedDict()
    num_started_points = 0
    # pbar = tqdm(total=num_start_points)
    # проверяем, что количество точек, по которым мы уже проитерировались, все еще меньше чем количество точек по
    # которым нас просят проитерироваться. или достаточно ли мы точек посетили, чтобы вернуть топ-К соседей(это
    # нужно в тех случаях, когда мы случайно могли первым выбором выбрать наиближайшего соседа, но все еще недобрали
    # информации по остальным соседям)
    while ((num_started_points < num_start_points) or (len(all_visited_points) < search_k)):
        # pbar.update(1)
        # выбираем случайный объект выборки
        cur_point_idx = np.random.randint(0, all_documents.shape[0] - 1)
        # проверяем на факт наличия рассчитанного расстояния для объекта в графе и если его нет рассчитываем расстояние
        cur_dist, verdict = calc_d_and_upd(all_visited_points, query_point, all_documents, cur_point_idx, dist_f)
        # verdict хранит в себе True|False и если значение уже было посчитано, то пропускаем эту итерацию
        if verdict:
            continue

        # Иначе запускаем цикл обхода всего графа
        while True:
            # Говорим, что минимальная дистанция на текущей итерации это текущая дистанция
            min_dist = cur_dist
            # выбранный кондидат это текущая точка, ведь возможно ничего лучше и нет
            choiced_cand = cur_point_idx

            # Далее для текущей точки с выбранным индексом достаем всех соседей построенных по ключу индекса
            cands_idxs = graph_edges[cur_point_idx]
            # Создаем множество кандидатов
            true_verdict_cands = set([cur_point_idx])
            # и начинаем по ним итерироваться
            for cand_idx in cands_idxs:
                # для каждого кондидата проверяем было ли найдено расстояние и если нет считаем его
                tmp_d, verdict = calc_d_and_upd(all_visited_points, query_point, all_documents, cand_idx, dist_f)
                # Смотрим, улучшилось ли наше приближение и если да, то обновляем минимальную дистанцию и обновляем
                # индекс кандидата, которого нам необходимо рассмотреть
                if tmp_d < min_dist:
                    min_dist = tmp_d
                    choiced_cand = cand_idx
                # если рассчет уже был, то мы просто эту точку добавляем
                if verdict:
                    true_verdict_cands.add(cand_idx)
            else:
                # если кондидат, в которого мы должны перейти уже в множестве точек, которые мы уже посчитали,
                # то заканчиваем итерацию
                if choiced_cand in true_verdict_cands:
                    break
                # обновляем точни и продолжаем итерироваться
                cur_dist = min_dist
                cur_point_idx = choiced_cand
                continue
        # увеличиваем счетчик точек, с которых мы начинали итерацию
        num_started_points += 1
    # отсортировали и берем лучших соседей
    best_idxs = np.argsort(list(all_visited_points.values()))[:search_k]
    final_idx = np.array(list(all_visited_points.keys()))[best_idxs]
    return final_idx

