import random
import numpy as np
from copy import deepcopy


def multiple_random_poisoning(data, X_test_list):
    X_train = data["X_train_clean"]
    y_train = data["y_train"]
    y_train_set = data["y_train_set"]

    assert len(X_test_list) > 1

    lower = np.zeros((len(X_test_list), len(y_train_set), len(X_train[0])), dtype=int)
    upper = np.zeros((len(X_test_list), len(y_train_set), len(X_train[0])), dtype=int)
    indicator = np.zeros((len(X_train), len(X_train[0])), dtype=int)
    label_count = np.zeros(len(y_train_set), dtype=int)

    for k in range(len(X_test_list)):
        X_test = X_test_list[k]
        for i in range(len(X_train)):
            for j in range(len(X_train[i])):
                if X_train[i][j] == X_test[j]:
                    lower[k][y_train[i]][j] += 1
                    upper[k][y_train[i]][j] += 1

    for i in range(len(y_train)):
        label_count[y_train[i]] += 1

    flag = True
    step = 0
    while flag:
        row = random.randint(0, len(X_train) - 1)
        col = random.randint(0, len(X_train[0]) - 1)
        if indicator[row][col] == 0:
            step += 1
            indicator[row][col] = 1
            for k in range(len(X_test_list)):
                X_test = X_test_list[k]
                if X_train[row][col] == X_test[col]:
                    lower[k][y_train[row]][col] -= 1
                else:
                    upper[k][y_train[row]][col] += 1

            flag_list = multiple_poisoning_decision(lower, upper, label_count, X_test_list)

            assert len(flag_list) == len(X_test_list)

            flag = False
            for fg in flag_list:
                if fg == True:
                    flag = True
                    break

    return step


def multiple_smarter_random_poisoning(data, X_test_list):
    X_train = data["X_train_clean"]
    y_train = data["y_train"]
    y_train_set = data["y_train_set"]

    lower = np.zeros((len(X_test_list), len(y_train_set), len(X_train[0])), dtype=int)
    upper = np.zeros((len(X_test_list), len(y_train_set), len(X_train[0])), dtype=int)
    pred_value = np.zeros((len(X_test_list), len(y_train_set)))
    label_count = np.zeros(len(y_train_set), dtype=int)

    for k in range(len(X_test_list)):
        X_test = X_test_list[k]
        for i in range(len(X_train)):
            for j in range(len(X_train[i])):
                if X_train[i][j] == X_test[j]:
                    lower[k][y_train[i]][j] += 1
                    upper[k][y_train[i]][j] += 1

    for i in range(len(y_train)):
        label_count[y_train[i]] += 1

    assert len(pred_value) == len(lower)
    for k in range(len(X_test_list)):
        for i in range(len(pred_value[k])):
            value = 1.0
            for j in range(len(lower[k][i])):
                value = value * lower[k][i][j] / label_count[i]
            pred_value[k][i] = value * label_count[i]

    total_op_dict = {}
    total_step = 0
    for k in range(len(X_test_list)):
        X_test = X_test_list[k]
        pred_label = np.argmax(pred_value[k])
        X_test_step = np.inf
        X_test_op_dict = {}
        for i in range(len(y_train_set)):
            if pred_label != y_train_set[i]:
                if pred_value[k][pred_label] > pred_value[k][y_train_set[i]]:
                    lower_real = deepcopy(lower[k][pred_label])
                    upper_real = deepcopy(upper[k][y_train_set[i]])
                    lower_real_value = 1.0
                    upper_real_value = 1.0
                    step = 0
                    op_dict = {}
                    assert len(lower_real) == len(upper_real)
                    for j in range(len(lower_real)):
                        lower_real_value = lower_real_value * lower_real[j] / label_count[pred_label]
                        upper_real_value = upper_real_value * upper_real[j] / label_count[y_train_set[i]]
                    lower_real_value = lower_real_value * label_count[pred_label]
                    upper_real_value = upper_real_value * label_count[y_train_set[i]]
                    while lower_real_value > upper_real_value:
                        if random.randint(0, 1) == 0:
                            col = random.randint(0, len(lower_real) - 1)
                            lower_real[col] -= 1
                            step += 1
                            lower_real_value = 1.0
                            for j in range(len(lower_real)):
                                lower_real_value = lower_real_value * lower_real[j] / label_count[pred_label]
                            lower_real_value = lower_real_value * label_count[pred_label]
                            if (col, X_test[col]) in op_dict.keys():
                                op_dict[(col, X_test[col])] += 1
                            else:
                                op_dict[(col, X_test[col])] = 1
                        else:
                            col = random.randint(0, len(lower_real) - 1)
                            upper_real[col] += 1
                            step += 1
                            upper_real_value = 1.0
                            for j in range(len(upper_real)):
                                upper_real_value = upper_real_value * upper_real[j] / label_count[y_train_set[i]]
                            upper_real_value = upper_real_value * label_count[y_train_set[i]]
                            if (col, X_test[col]) in op_dict.keys():
                                op_dict[(col, X_test[col])] += 1
                            else:
                                op_dict[(col, X_test[col])] = 1
                    if step < X_test_step:
                        X_test_step = step
                        X_test_op_dict = op_dict
                else:
                    raise ValueError("The probability of pred_label smaller than other labels")

        total_op_dict = merge_dict(total_op_dict, X_test_op_dict)
    for key, value in total_op_dict.items():
        total_step += value

    return total_step


def multiple_greedy_poisoning(data, X_test_list):
    X_train = data["X_train_clean"]
    y_train = data["y_train"]
    y_train_set = data["y_train_set"]

    lower = np.zeros((len(X_test_list), len(y_train_set), len(X_train[0])), dtype=int)
    upper = np.zeros((len(X_test_list), len(y_train_set), len(X_train[0])), dtype=int)
    pred_value = np.zeros((len(X_test_list), len(y_train_set)))
    label_count = np.zeros(len(y_train_set), dtype=int)

    for k in range(len(X_test_list)):
        X_test = X_test_list[k]
        for i in range(len(X_train)):
            for j in range(len(X_train[i])):
                if X_train[i][j] == X_test[j]:
                    lower[k][y_train[i]][j] += 1
                    upper[k][y_train[i]][j] += 1

    for i in range(len(y_train)):
        label_count[y_train[i]] += 1

    for k in range(len(X_test_list)):
        for i in range(len(pred_value[k])):
            value = 1.0
            for j in range(len(lower[k][i])):
                value = value * lower[k][i][j] / label_count[i]
            pred_value[k][i] = value * label_count[i]

    total_op_dict = {}
    total_step = 0
    for k in range(len(X_test_list)):
        X_test = X_test_list[k]
        pred_label = np.argmax(pred_value[k])
        X_test_step = np.inf
        X_test_op_dict = {}
        for i in range(len(y_train_set)):
            if pred_label != y_train_set[i]:
                if pred_value[k][pred_label] > pred_value[k][y_train_set[i]]:
                    lower_real = deepcopy(lower[k][pred_label])
                    upper_real = deepcopy(upper[k][y_train_set[i]])
                    pred_value_lower = pred_value[k][pred_label]
                    pred_value_upper = pred_value[k][y_train_set[i]]
                    lower_step = 0
                    lower_op_dict = {}
                    upper_step = 0
                    upper_op_dict = {}
                    while pred_value_lower > pred_value[k][y_train_set[i]]:
                        min_idx = np.argmin(lower_real)
                        lower_real[min_idx] -= 1
                        lower_step += 1
                        pred_value_lower = 1.0
                        for j in range(len(lower_real)):
                            pred_value_lower = pred_value_lower * lower_real[j] / label_count[pred_label]
                        pred_value_lower = pred_value_lower * label_count[pred_label]
                        if (min_idx, X_test[min_idx]) in lower_op_dict.keys():
                            lower_op_dict[(min_idx, X_test[min_idx])] += 1
                        else:
                            lower_op_dict[(min_idx, X_test[min_idx])] = 1

                    while pred_value_upper < pred_value[k][pred_label]:
                        min_idx = np.argmin(upper_real)
                        upper_real[min_idx] += 1
                        upper_step += 1
                        pred_value_upper = 1.0
                        for j in range(len(upper_real)):
                            pred_value_upper = pred_value_upper * upper_real[j] / label_count[y_train_set[i]]
                        pred_value_upper = pred_value_upper * label_count[y_train_set[i]]
                        if (min_idx, X_test[min_idx]) in upper_op_dict.keys():
                            upper_op_dict[(min_idx, X_test[min_idx])] += 1
                        else:
                            upper_op_dict[(min_idx, X_test[min_idx])] = 1

                    if lower_step < upper_step and X_test_step > lower_step:
                        X_test_op_dict = lower_op_dict
                        X_test_step = lower_step
                    elif upper_step < lower_step and X_test_step > upper_step:
                        X_test_op_dict = upper_op_dict
                        X_test_step = upper_step
                else:
                    raise ValueError("The probability of pred_label smaller than other labels")

        total_op_dict = merge_dict(total_op_dict, X_test_op_dict)
    for key, value in total_op_dict.items():
        total_step += value

    return total_step


def multiple_poisoning_decision(lower, upper, label_count, X_test_list):
    lower_list = np.zeros((len(X_test_list), len(lower[0])))
    upper_list = np.zeros((len(X_test_list), len(lower[0])))

    for k in range(len(X_test_list)):
        for i in range(len(lower[k])):
            lower_value = 1.0
            upper_value = 1.0
            for j in range(len(lower[k][i])):
                lower_value = lower_value * lower[k][i][j] / label_count[i]
                upper_value = upper_value * upper[k][i][j] / label_count[i]
            lower_list[k][i] = lower_value * label_count[i]
            upper_list[k][i] = upper_value * label_count[i]

    flag_list = []
    for k in range(len(X_test_list)):
        flag = False
        max_upper = -1
        max_upper_idx = -1
        second_max_upper = -1
        for i in range(len(upper_list[k])):
            if upper_list[k][i] > max_upper:
                max_upper = upper_list[k][i]
                max_upper_idx = i

        for i in range(len(upper_list[k])):
            if i != max_upper_idx and upper_list[k][i] > second_max_upper:
                second_max_upper = upper_list[k][i]

        if lower_list[k][max_upper_idx] > second_max_upper:
            flag = True
        flag_list.append(flag)

    return flag_list


def merge_dict(x, y):
    z = x.copy()
    for key in x.keys():
        if key in y.keys():
            z[key] = max(x[key], y[key])
        else:
            z[key] = x[key]

    for key in y.keys():
        if key in x.keys():
            z[key] = max(x[key], y[key])
        else:
            z[key] = y[key]
    return z
