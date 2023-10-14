import os
import sys
import random
import numpy as np
import time

from copy import deepcopy

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir)
    )
)

from tools.nbc import NaiveBayes


def approximate_decision(data, X_test_list):
    y_train = data["y_train"]
    col_dom = data["col_dom"]
    indicator = data["indicator"].values
    P = deepcopy(data["X_train_mv"])
    pred_list = []
    naivebayes = NaiveBayes()
    flag_list = []
    for i in range(len(X_test_list)):
        flag = True
        X_test = X_test_list[i]
        for k in range(100):
            for i in range(len(indicator)):
                for j in range(len(indicator[i])):
                    if indicator[i][j] == True:
                        P[i][j] = random.sample(col_dom[j], 1)[0]
            pred = naivebayes.predict_single(P, y_train, X_test)
            if len(pred_list) == 0:
                pred_list.append(pred)
            else:
                if pred_list[0] != pred:
                    flag = False
                    break
        flag_list.append(flag)


def iterative_algorithm(data, X_test_list):
    y_train = data["y_train"]
    y_train_set = data["y_train_set"]
    indicator = data["indicator"].values
    X_dirty = data["X_train_mv"]

    exact = np.zeros((len(X_test_list), len(y_train_set), len(X_dirty[0])), dtype=int)
    miss_dom = np.zeros((len(X_test_list), len(y_train_set), len(X_dirty[0])), dtype=int)
    label_count = np.zeros(len(y_train_set), dtype=int)
    lower_list = np.zeros(len(y_train_set), dtype=float)
    upper_list = np.zeros(len(y_train_set), dtype=float)

    for k in range(len(X_test_list)):
        X_test = X_test_list[k]
        for i in range(len(indicator)):
            for j in range(len(indicator[i])):
                if indicator[i][j] == True:
                    miss_dom[k][y_train[i]][j] += 1
                else:
                    if X_dirty[i][j] == X_test[j]:
                        exact[k][y_train[i]][j] += 1

    for i in range(len(y_train)):
        label_count[y_train[i]] += 1

    flag_list = []
    for k in range(len(X_test_list)):
        flag = False
        lower_matrix = exact[k]
        upper_matrix = exact[k] + miss_dom[k]

        for i in range(len(lower_matrix)):
            lower_value = 1.0
            upper_value = 1.0
            for j in range(len(lower_matrix[i])):
                lower_value = lower_value * lower_matrix[i][j] / label_count[i]
                upper_value = upper_value * upper_matrix[i][j] / label_count[i]
            lower_list[i] = lower_value * label_count[i]
            upper_list[i] = upper_value * label_count[i]

        max_upper = -1
        max_upper_idx = -1
        second_max_upper = -1

        for i in range(len(upper_list)):
            if upper_list[i] > max_upper:
                max_upper = upper_list[i]
                max_upper_idx = i

        for i in range(len(upper_list)):
            if i != max_upper_idx and upper_list[i] > second_max_upper:
                second_max_upper = upper_list[i]

        if lower_list[max_upper_idx] > second_max_upper:
            flag = True

        flag_list.append(flag)


def iterative_algorithm_with_index(data, X_test_list):
    y_train = data["y_train"]
    y_train_set = data["y_train_set"]
    indicator = data["indicator"].values
    X_dirty = data["X_train_mv"]

    value_dom = {}  # {col: {value: 0 1 2 ...}}
    count_array = np.zeros(len(X_dirty[0]), dtype=int)
    for i in range(len(X_dirty)):
        for j in range(len(X_dirty[i])):
            if j in value_dom.keys():
                if X_dirty[i][j] in value_dom[j].keys():
                    pass
                else:
                    value_dom[j][X_dirty[i][j]] = count_array[j]
                    count_array[j] += 1
            else:
                value_dom[j] = {}
                value_dom[j][X_dirty[i][j]] = count_array[j]

    exact = np.zeros((len(y_train_set), len(X_dirty[0]), int(max(count_array))), dtype=int)
    exact_comp = np.zeros((len(y_train_set), len(X_dirty[0])), dtype=int)
    miss_dom = np.zeros((len(y_train_set), len(X_dirty[0])), dtype=int)
    label_count = np.zeros(len(y_train_set), dtype=int)
    lower_list = np.zeros(len(y_train_set))
    upper_list = np.zeros(len(y_train_set))

    for i in range(len(indicator)):
        for j in range(len(indicator[i])):
            if indicator[i][j] == True:
                miss_dom[y_train[i]][j] += 1
            else:
                exact[y_train[i]][j][value_dom[j][X_dirty[i][j]]] += 1

    for i in range(len(y_train)):
        label_count[y_train[i]] += 1

    start_time = time.time()
    flag_list = []
    for k in range(len(X_test_list)):
        flag = False
        X_test = X_test_list[k]
        for i in range(len(y_train_set)):
            for j in range(len(X_dirty[0])):
                if X_test[j] not in value_dom[j].keys():
                    exact_comp[i][j] = 0
                else:
                    exact_comp[i][j] = exact[i][j][value_dom[j][X_test[j]]]
        lower_matrix = exact_comp
        upper_matrix = exact_comp + miss_dom

        for i in range(len(lower_matrix)):
            lower_value = 1.0
            upper_value = 1.0
            for j in range(len(lower_matrix[i])):
                lower_value = lower_value * lower_matrix[i][j] / label_count[i]
                upper_value = upper_value * upper_matrix[i][j] / label_count[i]
            lower_list[i] = lower_value * label_count[i]
            upper_list[i] = upper_value * label_count[i]
            # lower_list[i] = np.prod(lower_matrix[i]) / denominator[i]
            # upper_list[i] = np.prod(upper_matrix[i]) / denominator[i]

        max_upper = -1
        max_upper_idx = -1
        second_max_upper = -1

        for i in range(len(upper_list)):
            if upper_list[i] > max_upper:
                max_upper = upper_list[i]
                max_upper_idx = i

        for i in range(len(upper_list)):
            if i != max_upper_idx and upper_list[i] > second_max_upper:
                second_max_upper = upper_list[i]

        if lower_list[max_upper_idx] > second_max_upper:
            flag = True

        flag_list.append(flag)
    end_time = time.time()
    query_time = end_time - start_time

    return query_time

