# -*- coding: utf-8 -*-
"""
测试数据给了全量时序，这里就做离线的批量的异常检测
"""

import os
import copy


import numpy as np

from rca.ps import DataAPI
from settings import DIM_LIST, PT, LAYER, M, topN, Thresh_EP#, Thresh_EEP

Thresh_EEP = 0.10

Thresh_sur = 0.0001
# 要不要下钻分析取决于孩子门的异常一致性


def recursive_idice(dataAPI, predV=None, valueV=None):
    BSet_dict = dict()
    for layer in range(1, LAYER + 1):
        print("==================Begin search cuboids in layer %s======================" % layer)
        p_dims = load_cubes(layer, BSet_dict)
        BSet_dict[layer] = []
        if len(p_dims) <= 0:
            break
        for p_dim in p_dims:
            if not isinstance(p_dim[0], list):
                p_dim = [p_dim]
            explanatorySet = idice(dataAPI, p_dim=p_dim)
            if layer >= 2:
                BEST_BSet_j, BEST_ep, BEST_s, BEST_consistance = select_best_set(BSet_dict)
                try:
                    com_s = explanatorySet[0][-3]
                except:
                    com_s = 0
                if BEST_s <= com_s:
                    BSet_dict[layer].extend(explanatorySet)
            else:
                BSet_dict[layer].extend(explanatorySet)

        print("==================end search cuboids in layer %s======================" % layer)
        print(BSet_dict[layer])
    BEST_BSet_j, BEST_ep, BEST_score, BEST_consistance = select_best_set(BSet_dict)
    return (BEST_ep, BEST_consistance, BEST_BSet_j)


def select_best_set(BSet_dict):
    BEST_ep = -1
    BEST_BSet_j = None
    BEST_surp = -1
    BEST_consistance = 0
    BEST_s = 0
    for k, vS in BSet_dict.items():
        for v in vS:
            # if v[-2] > BEST_surp:
            f_s = v[1]
            if f_s > BEST_s:
                BEST_BSet_j = v[0]
                # BEST_surp = v[1]
                BEST_ep = v[2]
                BEST_consistance = v[3]
                BEST_s = f_s
    return (BEST_BSet_j, BEST_ep, BEST_s, BEST_consistance)


def f_score(p, q):
    score = 2.0 * p * 20 * q / (p * 20 + q)
    return score


def prepare_info_idice(dataAPI, p_dim):
    """
        [[[[0, 0, 5, 0, 0]], 0.0006747101914071126, 3.9397772622620613],
        [[[6, 0, 0, 0, 0]], 0.0005792226039303839, 3.6120409166158223]]
    return ：dim_list，可用于下钻一层的维度
    """
    valid_leaf = dataAPI.LEAF_DIMENSION_LIST
    valid_leaf = np.array(valid_leaf, dtype=int)
    # explanatorySet = []
    n_valid_leaf = valid_leaf.shape[0]
    p_dim_arr = np.array(p_dim)

    if p_dim_arr.size == 1:
        p_dim_arr = p_dim_arr[0]

    dim_list = []

    if isinstance(p_dim[0], list):
        tmp_p_dim = p_dim[0]
    else:
        tmp_p_dim = p_dim

    for i_th, tmp_p in enumerate(tmp_p_dim):
        if tmp_p == 0:
            dim_list.append(DIM_LIST[i_th])

    es_dim_list = []
    valid_dim_value_dim_list = []

    tmp_index = np.array([False] * n_valid_leaf, dtype=bool)
    valid_dim_value_org = np.zeros((n_valid_leaf, 5), dtype=int)
    ele_equal_Zero = np.where(p_dim_arr[0] == 0)[0]
    ele_not_equal_Zero = np.where(p_dim_arr[0] != 0)[0]
    if ele_not_equal_Zero.size > 0:
        valid_leaf_copy = copy.deepcopy(valid_leaf)
        for ele in p_dim_arr:
            # valid_dim = ele_not_equal_Zero
            valid_dim_array = valid_leaf_copy[:, ele_not_equal_Zero]
            tmp_index = tmp_index | np.all(valid_dim_array == ele[ele_not_equal_Zero], axis=1)

        for ind_ele_not_equal_Zero in ele_not_equal_Zero:
            valid_dim_value_org[tmp_index, ind_ele_not_equal_Zero] = copy.deepcopy(valid_leaf_copy[tmp_index, ind_ele_not_equal_Zero])

        valid_dim_value_org = valid_dim_value_org[tmp_index]
    else:
        tmp_index = np.array([True] * n_valid_leaf, dtype=bool)
        # valid_dim_value_org = valid_dim_value_org[tmp_index]
        pass

    # ele_not_equal_Zero
    # valid_leaf_slice = valid_leaf[tmp_index]
    # valid_dim_value_org = np.zeros((valid_leaf_slice, 5))
    # valid_dim_value_org[:, ele_equal_Zeor] = valid_leaf_slice[:, ele_equal_Zeor]

    for dim in dim_list:
        i_dim_ = int(dim.replace("dim", ""))
        valid_dim_value = copy.deepcopy(valid_dim_value_org)

        valid_dim_value[:, i_dim_ - 1] = copy.deepcopy(valid_leaf[tmp_index, i_dim_ - 1])

        valid_dim_value = np.unique(valid_dim_value, axis=0)

        es_list = []
        # candidateSet = []
        for v in valid_dim_value:
            if dataAPI.predict(v) < 10:
                es_list.append([0, 0, 0])
                continue
            ep_score = dataAPI.ep(v)

            if np.isnan(ep_score):
                ep_score = 0
            direct = 1 if dataAPI.total_delta > 0 else -1
            s_score = dataAPI.surprise(v, p_dim)
            if np.isnan(s_score):
                s_score = 0

            anomaly_score = dataAPI.value(v) * 1.0 / dataAPI.predict(v)
            if anomaly_score > 3:
                print("Notice!!!", anomaly_score)
            anomaly_score = min(anomaly_score, 5)

            if np.isnan(anomaly_score):
                anomaly_score = 0

            es_list.append([s_score, ep_score, anomaly_score])

        es_dim_list.append(es_list)
        valid_dim_value_dim_list.append(valid_dim_value)
    return (dim_list, es_dim_list, valid_dim_value_dim_list)


def idice(dataAPI, p_dim):
    dim_list, es_dim_list, valid_dim_value_dim_list = prepare_info_idice(dataAPI, p_dim)
    explanatorySet = []
    p_dim_ep = dataAPI.ep(p_dim)
    for i_dim, dim in enumerate(dim_list):
        es_list = es_dim_list[i_dim]
        valid_dim_value = valid_dim_value_dim_list[i_dim]

        es_arry = np.array(es_list, dtype=float)
        sorted_index = np.argsort(-es_arry[:, 1])
        ep_score_total = 0
        s_score_total = 0
        ep_score_total_org = 0
        candidateSet = []

        good_ind = es_arry[:, -1] > 0
        cnt = es_arry[good_ind, -1].size
        consistancy = np.nanstd(es_arry[good_ind, -1]) / np.sqrt(cnt)
        # anomaly_score_list = []
        for ind in sorted_index:
            [s_score, ep_score, anomaly_score] = es_arry[ind]
            if ep_score >= Thresh_EEP:
                candidateSet.append(valid_dim_value[ind].tolist())

                s_score_total += s_score
                ep_score_total_org += ep_score
            else:
                pass
            ep_score_total = ep_score_total_org # / p_dim_ep
            s_score = f_score(ep_score_total_org, consistancy)
            if ep_score_total > Thresh_EP or len(candidateSet) > 5:
                explanatorySet.append([candidateSet, s_score, ep_score_total_org, consistancy])
                break

    # replace surprise by consistance
    # surprise_arr = np.array([exp[-2] for exp in explanatorySet])
    # ind_sort = np.argsort(-surprise_arr)[:topN]
    consistance_arr = np.array([exp[-3] for exp in explanatorySet])
    ind_sort = np.argsort(-consistance_arr)[:topN]
    res = []
    for i in ind_sort:
        res.append(explanatorySet[i])
    return res


def load_cubes(layer=1, BSet_dict={}):
    if layer == 1:
        return [[0, 0, 0, 0, 0]]
    else:
        p_layer = layer - 1
        p_layer_set = BSet_dict.get(p_layer)
        if p_layer_set is None:
            return []
        p_elemsts = [i[0] for i in BSet_dict[p_layer]]
        # p_elemsts = np.array(p_elemsts, dtype=int).tolist()
        # import itertools
        # chain_list = list(itertools.chain(*p_elemsts))
        return p_elemsts


if __name__ == "__main__":
    error_ts = [1539567000000,
                1539909600000,
                1539948000000,
                1540532400000,
                1540534500000,                1540539600000,
                1540539600000,
                1540561200000]
    import time
    for ts in error_ts:
        begin = time.time()
        if ts != 1540539600000:
            continue
        dataAPI = DataAPI(ts=ts//1000, window=12)
        AA = recursive_idice(dataAPI)
        end = time.time()
        print("=======ts=======")
        print(ts, AA)
        print("====cost: %s====" % int(end - begin))








