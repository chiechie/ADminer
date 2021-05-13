# encoding=utf-8

from __future__ import print_function

import os
from os import listdir, mkdir
from os.path import join, abspath, dirname
from commons.path_ import mkdir_p

config_json = {
        #"LOCAL_UDF_DATA_DIR_shihuan": u"/Users/stellazhao/EasyML/data_asset/2019aiopsdata/",
        "LOCAL_UDF_DATA_DIR_shihuan": u"/Users/stellazhao/research_space/EasyML/data_asset/",

        "LOCAL_UDF_DATA_DIR_shuilian": u"/Users/shuiliantan/tencent_work/2019aiopsdata/",
        "REMOTE_DATA_DIR": u"/data/mapleleaf/shihuan/opsai-data/"}


def get_user_data_dir():
    if os.getcwd().startswith('/Users/stellazhao'):
        LOCAL_UDF_DATA_DIR = config_json.get('LOCAL_UDF_DATA_DIR_shihuan')
    if os.getcwd().startswith('/Users/shuiliantan'):
        LOCAL_UDF_DATA_DIR = config_json.get('LOCAL_UDF_DATA_DIR_shuilian')
    elif os.getcwd().startswith('/data/mapleleaf'):
        LOCAL_UDF_DATA_DIR = config_json.get('REMOTE_DATA_DIR')
    else:
        print(u'get_user_data_dir error ')
    print('your local data dir prefix is: %s' % (LOCAL_UDF_DATA_DIR))
    return LOCAL_UDF_DATA_DIR


local_root_dir = get_user_data_dir()
# 比赛共提供三份数据：
#第一次：
# Anomalytime_data_test1.csv
# 2019AIOps_data_test1/*.csv
#第二次：
# Anomalytime_data_test2.csv
# 2019AIOps_data_test2/*.csv
#第三次：
# Anomalytime_data_test3.csv
# 2019AIOps_data_test3/*.csv
####有标记的数据
#第四次


TEST_INDEX = 4
if TEST_INDEX <= 3:
#####输入#######
    label_path = join(local_root_dir, "2019aiopsdata/Anomalytime_data_test%d.csv" % TEST_INDEX)
    test_data_dir = join(local_root_dir, "2019aiopsdata/2019AIOps_data_test%d" % TEST_INDEX)
    test_data_path = listdir(test_data_dir)
else:
    label_path = join(local_root_dir, "rca_labeled_data/new_dataset_A_week_12_n_elements_1_layers_1/injection_info.csv")
    test_data_dir = join(local_root_dir, "rca_labeled_data/new_dataset_A_week_12_n_elements_1_layers_1/")
    test_data_path = [i for i in listdir(test_data_dir) if "injection_info" not in i]
    
    
DIM_LIST = ["dim1", "dim2", "dim3", "dim4", "dim5"]
multi_index = ["timestamp"] + DIM_LIST
SEED = 11111111

injection_config_small = dict(Random_Size=15,
    num_anomaly_in_each_layer={
    1: 5,
    2: 5,
    3: 5,
        })
injection_config = dict(Random_Size=300,
    num_anomaly_in_each_layer ={
    1: 100,
    2: 100,
    3: 100,
        })

chunksize = 10 ** 4

LAYER = 5

M = 200# 迭代次数

PT = 9.8

EPS = 1e-2
Eps = 1e-2
DEBUG = False


#####IDICE参数
# Thresh_EEP = 0.01
Thresh_EP = 0.9

topN = 5

######输入###########
dir_ = dirname(abspath(__file__))
#初始配置
DIM_DICT_path = join(dir_, "config/TEST_INDEX-%d/dim_dict_int.json" % TEST_INDEX)


## 中间数据和结果数据#######
dir_ = join(dir_, "tmp/TEST_INDEX-%d" % TEST_INDEX)
mkdir_p(dir_)


print("DIM_DICT_path", DIM_DICT_path)
offline_pred_LEAF_file = join(dir_, "offline_pred/%s-%s.csv")
# DIM_DICT = readJSON(DIM_DICT_path)
# DIM_DICT={}
# value_in_each_dim = list(DIM_DICT.values())

fake_anomaly_ts_file = join(dir_, "fake_ts/anomaly_ts_%s.csv")
fake_anomaly_set_file = join(dir_, "fake_ts_anomaly_set.csv")
fake_features_value_file = join(dir_, "fake_features/LEAF_value_ts_%s.pkl")
fake_features_pred_file = join(dir_, "fake_features/LEAF_pred_ts_%s.pkl")
fake_features_LEAF_file = join(dir_, "fake_features/LEAF_ts_%s.pkl")

real_features_value_file = join(dir_, "real_features/LEAF_value_ts_%s.pkl")
real_features_pred_file = join(dir_, "real_features/LEAF_pred_ts_%s.pkl")
real_features_LEAF_file = join(dir_, "real_features/LEAF_ts_%s.pkl")


real_hot_spot_result = join(dir_, "real_hot_spot_result/%s-%s.json")
formal_hot_spot_result = join(dir_, "real_hot_spot_result_formal_result.csv")
unformal_hot_spot_result = join(dir_, "real_hot_spot_result_unformal_result.csv")
real_dt_result = join(dir_, "real_dt_result/%s_%s.pkl")

real_algo_result = join(dir_, "real_%s_result/%s-%s.json")
formal_algo_result = join(dir_, "real_%s_result_formal_result.csv")
unformal_algo_result = join(dir_, "real_%s_result_unformal_result.csv")


fake_dt_result = join(dir_, "fake_dt_result/%s_%s.pkl")
fake_hot_spot_result = join(dir_, "fake_hot_spot_result/%s-%s.json")

fake_hot_spot_pred_result_path = join(dir_, "fake_hot_spot_pred_result.csv")
fake_hot_spot_res_dict_path = join(dir_, "fake_hot_spot_evaluation.json")
fake_hot_spot_real_result_path = join(dir_, "fake_hot_spot_real_result.csv")
# real_features_contri_file = join(dir_, "real_features/contri_ts_%s.pkl")
# real_features_detect_file = join(dir_, "real_features/detect_ts_%s.pkl")
