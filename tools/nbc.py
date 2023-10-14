import numpy as np


class NaiveBayes(object):
    def __init__(self):
        super(NaiveBayes).__init__()
        # {column: {dom: {label: num} } }
        self.feature_count = {}
        # {label: num}
        self.label_count = {}

    def fit(self, X_train, y_train):
        for i in range(len(y_train)):
            if y_train[i] in self.label_count.keys():
                self.label_count[y_train[i]] += 1
            else:
                self.label_count[y_train[i]] = 1

        for i in range(len(X_train)):
            for j in range(len(X_train[i])):
                if j in self.feature_count.keys():
                    if X_train[i][j] in self.feature_count[j].keys():
                        if y_train[i] in self.feature_count[j][X_train[i][j]].keys():
                            self.feature_count[j][X_train[i][j]][y_train[i]] += 1
                        else:
                            self.feature_count[j][X_train[i][j]][y_train[i]] = 1
                    else:
                        self.feature_count[j][X_train[i][j]] = {y_train[i]: 1}
                else:
                    self.feature_count[j] = {X_train[i][j]: {y_train[i]: 1}}

    def predict(self, X_test):
        pred_dict = {}
        for key in self.label_count.keys():
            value = 1 / (self.label_count[key] ** (len(X_test) - 1))
            for i in range(len(X_test)):
                if i in self.feature_count.keys():
                    if X_test[i] in self.feature_count[i].keys():
                        if key in self.feature_count[i][X_test[i]].keys():
                            value = value * self.feature_count[i][X_test[i]][key]
                        else:
                            value = 0
                    else:
                        value = 0
                else:
                    value = 0
            pred_dict[key] = value
        max_value = 0
        max_key = ""
        for key in pred_dict.keys():
            if pred_dict[key] > max_value:
                max_value = pred_dict[key]
                max_key = key
        return max_key

    def predict_single(self, X_train, y_train, X_test):
        pred_dict = {}
        label_count = {}
        feature_count = {}
        for i in range(len(y_train)):
            if y_train[i] in label_count.keys():
                label_count[y_train[i]] += 1
            else:
                label_count[y_train[i]] = 1

        for key in label_count.keys():
            feature_count[key] = np.zeros(len(X_train[0]))

        for i in range(len(X_train)):
            for j in range(len(X_train[i])):
                if X_train[i][j] == X_test[j]:
                    feature_count[y_train[i]][j] += 1

        for key in label_count.keys():
            value = 1
            for i in range(len(feature_count[key])):
                value = value * feature_count[key][i] / (1.0 * label_count[key])
            pred_dict[key] = value * 1.0 * label_count[key]
            # pred_dict[key] = value / (1.0 * label_count[key] ** (len(feature_count[key]) - 1))

        max_value = 0
        max_key = ""
        for key in pred_dict.keys():
            if pred_dict[key] > max_value:
                max_value = pred_dict[key]
                max_key = key
        return max_key

