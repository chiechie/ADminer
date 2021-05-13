# encoding=utf-8
from __future__ import print_function
import re
import os
from os import listdir
from os.path import join
import json
from itertools import combinations, product

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.utils.fixes import signature

from commons.date_ import format_to_millisec, millisec_to_str
from commons.path_ import readJSON, readDF
from rca.settings import  TEST_INDEX,DIM_LIST, DIM_DICT_path, test_data_dir, test_data_path, label_path, offline_pred_LEAF_file
from rca.util import encode_dim_value, decode_dim_value

start_ts0=format_to_millisec("2018-09-01 00:00:00")
end_ts0=format_to_millisec("2018-09-14 23:55:00")


DATA_CACHE = {}
def set_Cache(k, v):
    global DATA_CACHE
    DATA_CACHE[k] = v
    return

def get_Cache(k):
    global DATA_CACHE
    v = DATA_CACHE.get(k)
    return v


def load_cuboids(layer=2, valid_dim_value=None):
    # dim_list = list(dim_dict.keys())
    cuboids_elements = get_Cache("cuboids_layer%s"% layer)
    if cuboids_elements is not None:
        return cuboids_elements

    DIM_DICT = readJSON(DIM_DICT_path)
    cuboids = list(combinations(DIM_LIST, layer))
    cuboids_elements = list()
    for cuboid in cuboids:
        if valid_dim_value is not None:
            value_in_each_dim = [list(set(DIM_DICT[d])&set(valid_dim_value[d])) for d in cuboid]
        else:
            value_in_each_dim = [DIM_DICT[d] for d in cuboid]
        cuboid_LEAF = list(product(*value_in_each_dim))

        cuboid_LEAF_index = [np.argwhere(np.array(DIM_LIST) == i)[0][0] for i in cuboid]
        leaf_elements = np.zeros(shape=(len(cuboid_LEAF), len(DIM_LIST)), dtype=int)
        leaf_elements[:, cuboid_LEAF_index] = cuboid_LEAF
        #leaf_elements[leaf_elements == 0] = None
        # cuboid_ele = [encode_dim_value(ele_list) for ele_list in cuboid_LEAF]
        cuboids_elements.append(leaf_elements.tolist())
    set_Cache("cuboids_layer%s" % layer, cuboids_elements)
    return cuboids_elements


def plot_layer1_cuboids(cuboid_1_dim):
    for dim_name, dim_set in cuboid_1_dim.iteritems():
        plt.close()
        for d, v in dim_set.iteritems():
            v.name = d
            v.plot()
        plt.legend()
        plt.show()
    return None


def load_anomaly_ts(label_path=label_path):
    print("label_path",label_path)
    #label_df = readDF(label_path , header=None, names=["timestamp"])
    label_df = readDF(label_path)
    label_df["timestamp"] = label_df["timestamp"].astype(np.int)
    return label_df


def load_test_data(start_str="2018-09-15 00:00:00", end_str="2018-09-29 00:00:00", dimension={},  fast_mode=False):
    valid_test_data_path = load_valid_data_path(start_str, end_str, fast_mode)
    print("[readDF]TOTAL:%s" % (len(valid_test_data_path)))
    total_df_list = []

    for t in valid_test_data_path:
        if len(dimension) < 1:
            # print("[readDF]%s" % join(test_data_dir, t))
            if TEST_INDEX <= 3:
                tmp = readDF(join(test_data_dir, t), header=None, names=DIM_LIST + ["value"])
            else:
                tmp = readDF(join(test_data_dir, t))
                tmp.rename(columns={"real": "value",
                                    "i": "dim1",
                                    "e": "dim2",
                                    "c": "dim3",
                                    "p": "dim4",
                                    "l": "dim5",
                                  }, inplace=True)

            tmp = valid(tmp)
            # tmp = tmp[tmp__mask]
        else:
            # print("[readDF]%s" % join(test_data_dir, t))
            chunks = pd.read_csv(join(test_data_dir, t), chunksize=chunksize, header=None, names=DIM_LIST  + ["value"])
            # chunks.columns = dim_list + ["value"]
            dim = list(dimension)[0]
            dim_value = dimension[dim]
            tmp = pd.concat(valid(chunks, dim, dim_value))

        tmp = tmp[tmp['value'] > 0]
        # tmp["timestamp"] = millisec_to_str(int(t.split(".csv")[0]) / 1000)
        ts_str = t.split(".csv")[0]
        if len(ts_str) > 10:
            tmp.insert(0, 'timestamp', int(ts_str) / 1000)
        else:
            tmp.insert(0, 'timestamp', int(ts_str) )

        total_df_list.append(tmp)

    total_df = pd.concat(total_df_list, axis=0)
    total_df = map_str_dimvalue_2_int(total_df)
    total_df.sort_values("timestamp", inplace=True)
    return total_df


def map_str_dimvalue_2_int(df):
    str2int = lambda a: int(re.findall(r"\d+", a)[0])
    for col in DIM_LIST:
        df[col] = df[col].map(str2int)
    return df


def load_valid_data_path(start_str="2018-09-15 00:00:00", end_str="2018-09-29 00:00:00", fast_mode=False):
    start_sec = format_to_millisec(start_str)
    end_sec = format_to_millisec(end_str)
    valid_test_data_path = []
    for ts in test_data_path:
        if ".csv" not in ts:
            continue
        ts_str = ts.split(".csv")[0]
        if len(ts_str) > 10:
            tmp_ts = int(ts_str) / 1000
        else:
            tmp_ts = int(ts_str)

        if fast_mode and tmp_ts % 600 != 0:
            continue

        if tmp_ts >= start_sec and tmp_ts < end_sec:
            valid_test_data_path.append(ts)
    return valid_test_data_path


def valid(chunk, dim=None, dim_value=None):
    if dim is None:
        mask0 = pd.Series([True] * chunk.shape[0])
    else:
        mask0 = chunk[dim] == dim_value
    mask1 = ~chunk.dim1.str.contains('unknown')
    mask2 = ~chunk.dim2.str.contains('unknown')
    mask3 = ~chunk.dim3.str.contains('unknown')
    mask4 = ~chunk.dim4.str.contains('unknown')
    mask5 = ~chunk.dim5.str.contains('unknown')
    mask = mask0 & mask1 & mask2 & mask3 & mask4 & mask5
    data_mask = chunk[mask]
    return data_mask


def load_valid_LEAF(ts=None, window=3, data=None):
    if data is None:
        start_sec = format_to_millisec(ts) - 60 * 60 * window + 300
        start_str = millisec_to_str(start_sec)
        data = load_test_data(start_str=start_str, end_str=ts, dimension={})
    data = data[data['value'] > 0]
    dim_array = data[DIM_LIST].values
    valid_dim_int_list = np.unique(dim_array, axis=0).tolist()
    return valid_dim_int_list




def load_leaf_f_vector(ts, batch_size = 1):
    file_index = (ts - start_ts0) // int(batch_size * 24 * 3600)
    start_ts = start_ts0 + int(file_index * batch_size * 24 * 3600)
    end_ts = start_ts + int(batch_size * 24 * 3600)
    file_name = offline_pred_LEAF_file % (start_ts, end_ts)
    offline_pred_df = readDF(file_name)
    slice_df = offline_pred_df[offline_pred_df["timestamp"] == ts]
    return slice_df



if __name__ == "__main__":
    cuboids_elements = load_cuboids(1)
    print("cuboids_elements", cuboids_elements)