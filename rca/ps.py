# encoding=utf-8
from collections import Iterable
import copy
import time
import numpy as np
import pandas as pd
import numpy_indexed as npi
from statsmodels.nonparametric.smoothers_lowess import lowess

def mask(df, key, value):
    return df[df[key].values == value]

pd.DataFrame.mask = mask

from commons.date_ import format_to_millisec, millisec_to_str
from commons.path_ import readJSON, setCache, getCache
from commons.matrix_helper import resample

from rca.load_data import load_test_data, load_valid_LEAF, load_leaf_f_vector, DATA_CACHE, get_Cache, set_Cache
from rca.settings import EPS, DIM_LIST, offline_pred_LEAF_file
from rca.util import encode_dim_value, decode_dim_value, add_dim_info
from rca.util import idiceintersect
"""
对于所有LEAF构成的vector的操作
"""

# value_in_each_dim = list(DIM_DICT.values())
class DataAPI(object):

    def __init__(self, ts, window=1, LEAF_DIMENSION_LIST=[], data=None, fast_mode=False):
        """
        :param ts:
        :param window: int类型，单位为小时
        :param LEAF_DIMENSION_LIST:
        """
        self.ts = ts
        self.window = window
        self.start = ts - window * 60 * 60 + 5 * 60
        self.timestamp_list = list(self.start + np.arange(window * 12) * 60 * 5)
        # begin = time.time()
        if data is None:
            # 加载数据的时候要注意，这里是左闭右开的逻辑
            self._data = load_test_data(start_str=millisec_to_str(self.start), end_str=millisec_to_str(ts + 300) , dimension={}, fast_mode=fast_mode)
        else:
            # 这个是dataframe哦
            self._data = data
        self._data["timestamp"].astype = int
        self._data[DIM_LIST].astype = int

        # end = time.time()
        # print("[Endload_test_data@%s]cost %s secs" % (ts, int(end - begin)))

        self.total_value = self.value([0, 0, 0, 0, 0], window=False)
        self.total_predict = self.predict([0, 0, 0, 0, 0], window=False)
        self.total_delta = self.total_predict - self.total_value
        if LEAF_DIMENSION_LIST == []:
            self.LEAF_DIMENSION_LIST = load_valid_LEAF(data=self._data, window=window)
        self.valid_dim_str_list = LEAF_DIMENSION_LIST

    def get_LEAF_v_vector0(self):
        print("[BeginComputingVvector]@%s.It can be consuming" % self.ts)
        v_vector = [self.value(i) for i in self.LEAF_DIMENSION_LIST]
        return v_vector

    def get_LEAF_f_vector0(self):
        print("[BeginComputingFvector]@%s.It can be consuming" % self.ts)
        f_vector = [self.predict(i) for i in self.LEAF_DIMENSION_LIST]
        return f_vector


    def get_LEAF_f_vector(self):
        print("[BeginComputingFvector]@%s.It can be consuming" % self.ts)
        df_slice = load_leaf_f_vector(self.ts, 0.5)
        print("[get_leaf_f_vector res]", df_slice.shape)
        # df_sclice_dim = df_slice[DIM_LIST].value
        # df_sclice_value = df_slice["prediction"].value
        # idiceintersect(self.LEAF_DIMENSION_LIST, df_sclice_value)

        ts_array = df_slice["timestamp"].values
        dim_array = df_slice[DIM_LIST].values
        value_array = df_slice["predition"].values
        f_vector = []
        for element in self.LEAF_DIMENSION_LIST:
            slice_data_array = sclice_dim_value(element, ts_array, dim_array, value_array)
            if slice_data_array is None:
                f_vector.append(0)
            else:
                f_vector.append(slice_data_array[:, 1].sum())
        return f_vector

    def get_LEAF_v_vector(self):
        print("[BeginComputingVvector]@%s.It can be consuming" % self.ts)
        df_slice = load_leaf_f_vector(self.ts, 0.5)
        print("[get_leaf_v_vector shape ]", df_slice.shape)
        ts_array = df_slice["timestamp"].values
        dim_array = df_slice[DIM_LIST].values
        value_array = df_slice["value"].values

        v_vector = []
        for element in self.LEAF_DIMENSION_LIST:
            slice_data_array = sclice_dim_value(element, ts_array, dim_array, value_array)
            if slice_data_array is None:
                v_vector.append(0)
            else:
                v_vector.append(slice_data_array[:, 1].sum())

        return v_vector

    def ps_value(self, S_dimension, predV, valueV):
        """
        S_dimension = [6, 0, 5 , 0, 0]

        """
        leaf_dim_array = np.array(self.LEAF_DIMENSION_LIST)
        S_dimension = np.array(S_dimension, dtype=int).reshape(-1, 5)
        # S_dimension = S_dimension.view([('', S_dimension.dtype)] * S_dimension.shape[1]).ravel()
        # if S_dimension[0] == 42:
        #     print("dede")

        N_leaf_dim = leaf_dim_array.shape[0]
        S_dimension_pred, _, S_dimension_value = self.predict(S_dimension, vol=True)
        h_S = S_dimension_pred - S_dimension_value
        if S_dimension_pred < 1:
            return 0
        # get A_Vector
        A_vector = np.zeros(N_leaf_dim)
        S_dimension = np.array(S_dimension, dtype=int).ravel()
        n_S_dimension = S_dimension.shape[0]
        if n_S_dimension > 1:
            valid_dim = np.where(S_dimension[0] != 0)[0]
        else:
            valid_dim = np.where(S_dimension != 0)[0]
        valid_data_array = leaf_dim_array[:, valid_dim]
        s_child_leaf_ind = np.all(valid_data_array == S_dimension[valid_dim], axis=1)
        A_vector[s_child_leaf_ind] = predV[s_child_leaf_ind] - 1.0 * h_S * predV[s_child_leaf_ind] / max(S_dimension_pred, EPS)
        #revised
        # A_vector[s_child_leaf_ind] = predV[s_child_leaf_ind] - 1.0 * h_S / predV[s_child_leaf_ind].size

        dist_v_a = np.sqrt(np.linalg.norm(valueV - A_vector))
        dist_v_f = np.sqrt(np.linalg.norm(valueV - predV))

        ps = max(10 - dist_v_a / dist_v_f, 0)
        # print("[GetPS]", S_dimension, ps)
        return ps

    def get_a_value(self, root_set, leaf_ele, h_S):
        a_value = self.predict(leaf_ele) - 1.0 * h_S * self.predict(leaf_ele) / max(self.predict(root_set), EPS)
        return a_value

    def predict(self, s_dimension_value, vol=False, window=3, frac=0.2):
        k = "predict/@%s/%s%d" % (self.ts, s_dimension_value, window)
        res_p = get_Cache(k)
        if res_p is not None:
            return res_p

        s_dimension_value = np.array(s_dimension_value, dtype=int).reshape(-1, 5)
        ser = self.value(s_dimension_value, window=True)
        # if ser.empty:
        if ser.size == 0:
            pred, stand, real = 0, 0, 0
        else:
            ser_len = ser.shape[0]
            # win_size = min(window+1, ser_len)
            x = np.arange(ser_len).tolist()
            y = ser[:,1]

            filtered = lowess(y, x, is_sorted=True, frac=frac, it=2)
            pred = filtered[:, 1][-1]
            real = ser[-1, 1]

            if ser_len <= 1:
                stand = 0
            else:
                # stand = ser.std()
                stand = np.std(filtered[:, 1])
        if not vol:
            res_pred = pred
        else:
            # print("pred, stand", pred, stand)
            res_pred = (pred, stand, real)
        set_Cache(k, res_pred)
        return res_pred

    def consist(self, s_dimension_value, predict_Vector, value_vector):
        """
        衡量s_dimension_value的子节点的一致性,
        返回为float类型，取值越大，说明越一致
        """
        leaf_array = np.array(self.LEAF_DIMENSION_LIST)
        consist_array = self.get_child_leaf(s_dimension_value)
        delta_child = (predict_Vector[consist_array] - value_vector[consist_array]) / predict_Vector[consist_array] 
        consist_value = np.nanmean(delta_child) /np.nanstd(delta_child)
        return consist_value

    def ep(self, s_dimension_value):
        """
        衡量s_dimension_value的异常贡献度
        返回向量
        """
        # total_delta = np.sum(predict_Vector) - np.sum(value_vector)

        frac1 = self.predict(s_dimension_value, frac=0.2) - self.value(s_dimension_value) 
        frac2 = self.total_delta
        # frac2 = self.predict(p_dim, frac=0.2) - self.value(p_dim)
        #     if frac1 > frac2:
        #         print("s_dimension_value", s_dimension_value)
        #         print("frac1", frac1)
        #         print("frac2", frac2)
        anomaly_ratio_contribute_value = frac1 * 1.0 / frac2

        return anomaly_ratio_contribute_value

    def surprise(self, v, p_dim):
        p_predict = self.predict(v) / self.predict(p_dim)
        q_value = self.value(v) / self.value(p_dim)
        s = kl_point1(p_predict, q_value)
        return s

    def get_child_leaf(self, s_dimension_value):
        leaf_array = np.array(self.LEAF_DIMENSION_LIST)
        consist_array = np.array([1] * len(self.LEAF_DIMENSION_LIST), dtype=bool)

        for i, s in enumerate(s_dimension_value):
            if s == 0:
                continue
            else:
                consist_array = (consist_array) & (leaf_array[:, i] == s) 
        return consist_array


    def get_LEAF_contribution_vector(self):
        LEAF_DIMENSION_LIST = self.LEAF_DIMENSION_LIST
        k = "%s/contribution" % (self.ts)
        v_vector = get_Cache(k)
        if v_vector is None:
            v_vector = [self.contribution(i) for i in LEAF_DIMENSION_LIST]
            set_Cache(k, v_vector)
        assert len(LEAF_DIMENSION_LIST) == len(v_vector)
        return v_vector

    def get_LEAF_detect_vector(self):
        LEAF_DIMENSION_LIST = self.LEAF_DIMENSION_LIST
        d_vector = [self.detect(i) for i in LEAF_DIMENSION_LIST]
        assert len(LEAF_DIMENSION_LIST) == len(d_vector)
        return d_vector

    def contribution(self, s_dimension_value):
        ser = self.value(s_dimension_value, window=True)
        # if ser.empty:
        if ser.size == 0:
            return 0
        # ser_mean = ser.mean()
        ser_mean = np.mean(ser[:, 1])
        return ser_mean

    def detect(self, s_dimension_value):

        mean, std, real = self.predict(s_dimension_value, vol=True)

        if abs(std) <= EPS * 0.0001:
            return 0

        z_score = (real - mean) / std
        return z_score

    def value(self, element, window=False):
        """
        element: element是由int构成的list，如[1,1,1,1,1]
        """
        k = "value/@%s/%s%d" % (self.ts, element, window)
        res_v = get_Cache(k)
        if res_v is not None:
            return res_v

        data_array = self._data.values
        n_samples, _ = data_array.shape
        #这里要小心，value可能是整数
        # data_array = np.array(data_array, dtype=int)
        ts_array = self._data["timestamp"].values
        dim_array = self._data[DIM_LIST].values
        value_array = self._data["value"].values

        slice_data_array = sclice_dim_value(element, ts_array, dim_array, value_array)

        if slice_data_array is None:
            slice_data_array = np.array([self.ts, 0]).reshape(-1, 2)

        ser = resample_array(slice_data_array, self.timestamp_list)
        if window:
            res = ser
        else:
            res = ser[-1, 1]
        set_Cache(k, res)
        return res


def kl_point1(p,q):
    # p: predict
    # q: value
    s = 0.5 * (p * np.log(2.0 * p / (p + q)) + q * np.log( 2.0 * q  / (p + q)))
    return s


def kl_point(p,q):
    # p: predict
    # q: value
    sub = p - q
    return sub


def sclice_dim_value(element, ts_array, dim_array, value_array):
    element = np.array(element, dtype=int).reshape(-1, 5)

    valid_dim_ind = np.where(element != 0)[0]
    n_samples = ts_array.shape[0]

    if valid_dim_ind.size == 0:
        slice_data_array = np.concatenate((ts_array.reshape(-1, 1), value_array.reshape(-1, 1)), axis=1)
    else:
        tmp_index = np.array([False] * n_samples, dtype=bool)
        for ele in element:
            valid_dim = np.where(ele != 0)[0]
            valid_dim_array = dim_array[:, valid_dim]
            tmp_index = tmp_index | np.all(valid_dim_array == ele[valid_dim], axis=1)

        slice_value_array = value_array[tmp_index]
        slice_ts_array = ts_array[tmp_index]

        if slice_value_array.size == 0:
            slice_data_array = None
        else:
            slice_data_array = np.concatenate((slice_ts_array.reshape(-1, 1), slice_value_array.reshape(-1, 1)), axis=1)
    return slice_data_array



def resample_array(data_mask, timestamp_list):
    groups = data_mask[:, 0]
    groups = np.array(groups, dtype=int)
    data = data_mask[:, 1]
    ts_list = np.array(np.unique(groups), dtype=int).tolist()

    value_list = [data[groups == ts_].sum() for ts_ in ts_list]

    res_array = np.zeros((len(timestamp_list), 2), dtype=float)
    res_array[:, 0] = timestamp_list

    match_index = npi.indices(timestamp_list, ts_list)
    res_array[match_index, 1] = value_list
    return res_array


if __name__ == "__main__":
    pass



