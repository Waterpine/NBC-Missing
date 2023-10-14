import random
import numpy as np
from copy import deepcopy


def random_poisoning(data, X_test):
    X_train = data["X_train_clean"]
    y_train = data["y_train"]
    y_train_set = data["y_train_set"]

    lower = np.zeros((len(y_train_set), len(X_train[0])), dtype=int)
    upper = np.zeros((len(y_train_set), len(X_train[0])), dtype=int)
    indicator = np.zeros((len(X_train), len(X_train[0])), dtype=int)
    label_count = np.zeros(len(y_train_set), dtype=int)

    for i in range(len(X_train)):
        for j in range(len(X_train[i])):
            if X_train[i][j] == X_test[j]:
                lower[y_train[i]][j] += 1
                upper[y_train[i]][j] += 1

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
            if X_train[row][col] == X_test[col]:
                lower[y_train[row]][col] -= 1
            else:
                upper[y_train[row]][col] += 1

            flag = poisoning_decision(lower, upper, label_count)

    return step


def smarter_random_poisoning(data, X_test):
    X_train = data["X_train_clean"]
    y_train = data["y_train"]
    y_train_set = data["y_train_set"]

    lower = np.zeros((len(y_train_set), len(X_train[0])), dtype=int)
    upper = np.zeros((len(y_train_set), len(X_train[0])), dtype=int)
    pred_value = np.zeros(len(y_train_set))
    label_count = np.zeros(len(y_train_set), dtype=int)

    for i in range(len(X_train)):
        for j in range(len(X_train[i])):
            if X_train[i][j] == X_test[j]:
                lower[y_train[i]][j] += 1
                upper[y_train[i]][j] += 1

    for i in range(len(y_train)):
        label_count[y_train[i]] += 1

    assert len(pred_value) == len(lower)
    for i in range(len(pred_value)):
        value = 1.0
        for j in range(len(lower[i])):
            value = value * lower[i][j] / label_count[i]
        pred_value[i] = value * label_count[i]

    pred_label = np.argmax(pred_value)

    step_list = []
    for i in range(len(y_train_set)):
        if pred_label != y_train_set[i]:
            step = 0
            if pred_value[pred_label] > pred_value[y_train_set[i]]:
                lower_real = deepcopy(lower[pred_label])
                upper_real = deepcopy(upper[y_train_set[i]])
                lower_real_value = 1.0
                upper_real_value = 1.0
                assert len(lower_real) == len(upper_real)
                for j in range(len(lower_real)):
                    lower_real_value = lower_real_value * lower_real[j] / label_count[pred_label]
                    upper_real_value = upper_real_value * upper_real[j] / label_count[y_train_set[i]]
                lower_real_value = lower_real_value * label_count[pred_label]
                upper_real_value = upper_real_value * label_count[y_train_set[i]]
                # while np.prod(lower_real) / denominator[pred_label] > np.prod(upper_real) / denominator[y_train_set[i]]:
                while lower_real_value > upper_real_value:
                    if random.randint(0, 1) == 0:
                        col = random.randint(0, len(lower_real) - 1)
                        lower_real[col] -= 1
                        step += 1
                        lower_real_value = 1.0
                        for j in range(len(lower_real)):
                            lower_real_value = lower_real_value * lower_real[j] / label_count[pred_label]
                        lower_real_value = lower_real_value * label_count[pred_label]
                    else:
                        col = random.randint(0, len(lower_real) - 1)
                        upper_real[col] += 1
                        step += 1
                        upper_real_value = 1.0
                        for j in range(len(upper_real)):
                            upper_real_value = upper_real_value * upper_real[j] / label_count[y_train_set[i]]
                        upper_real_value = upper_real_value * label_count[y_train_set[i]]
            else:
                raise ValueError("The probability of pred_label smaller than other labels")
            step_list.append(step)

    return min(step_list)


def greedy_poisoning(data, X_test):
    X_train = data["X_train_clean"]
    y_train = data["y_train"]
    y_train_set = data["y_train_set"]

    lower = np.zeros((len(y_train_set), len(X_train[0])), dtype=int)
    upper = np.zeros((len(y_train_set), len(X_train[0])), dtype=int)
    pred_value = np.zeros(len(y_train_set))
    label_count = np.zeros(len(y_train_set), dtype=int)

    for i in range(len(X_train)):
        for j in range(len(X_train[i])):
            if X_train[i][j] == X_test[j]:
                lower[y_train[i]][j] += 1
                upper[y_train[i]][j] += 1

    for i in range(len(y_train)):
        label_count[y_train[i]] += 1

    for i in range(len(pred_value)):
        value = 1.0
        for j in range(len(lower[i])):
            value = value * lower[i][j] / label_count[i]
        pred_value[i] = value * label_count[i]

    pred_label = np.argmax(pred_value)

    step_list = []
    for i in range(len(y_train_set)):
        if pred_label != y_train_set[i]:
            if pred_value[pred_label] > pred_value[y_train_set[i]]:
                lower_real = deepcopy(lower[pred_label])
                upper_real = deepcopy(upper[y_train_set[i]])
                pred_value_lower = pred_value[pred_label]
                pred_value_upper = pred_value[y_train_set[i]]
                lower_steps = 0
                upper_steps = 0
                while pred_value_lower > pred_value[y_train_set[i]]:
                    min_idx = np.argmin(lower_real)
                    lower_real[min_idx] -= 1
                    lower_steps += 1
                    pred_value_lower = 1.0
                    for j in range(len(lower_real)):
                        pred_value_lower = pred_value_lower * lower_real[j] / label_count[pred_label]
                    pred_value_lower = pred_value_lower * label_count[pred_label]

                while pred_value_upper < pred_value[pred_label]:
                    min_idx = np.argmin(upper_real)
                    upper_real[min_idx] += 1
                    upper_steps += 1
                    pred_value_upper = 1.0
                    for j in range(len(upper_real)):
                        pred_value_upper = pred_value_upper * upper_real[j] / label_count[y_train_set[i]]
                    pred_value_upper = pred_value_upper * label_count[y_train_set[i]]

                step_list.append(min(lower_steps, upper_steps))

            else:
                raise ValueError("The probability of pred_label smaller than other labels")

    return min(step_list)


def poisoning_decision(lower, upper, label_count):
    flag = False
    lower_list = np.zeros(len(lower))
    upper_list = np.zeros(len(lower))

    for i in range(len(lower)):
        lower_value = 1.0
        upper_value = 1.0
        for j in range(len(lower[i])):
            lower_value = lower_value * lower[i][j] / label_count[i]
            upper_value = upper_value * upper[i][j] / label_count[i]
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

    return flag


