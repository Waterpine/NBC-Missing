import os
import numpy as np
import pandas as pd

from copy import deepcopy
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class Preprocessor(object):
    """docstring for Preprocessor"""
    def __init__(self, num_strategy="median"):
        super(Preprocessor, self).__init__()
        self.num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=num_strategy)),
            ('scaler', MinMaxScaler()),
            ('kbins', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform'))
        ])
        self.cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
        self.label_enc = LabelEncoder()

    def fit(self, X_train, y_train):
        self.num_features = X_train.select_dtypes(include='number').columns
        self.cat_features = X_train.select_dtypes(exclude='number').columns
        self.features_index = []
        for col in self.num_features.tolist():
            self.features_index.append(X_train.columns.get_loc(col))

        for col in self.cat_features.tolist():
            self.features_index.append(X_train.columns.get_loc(col))

        if len(self.num_features) > 0:
            self.num_transformer.fit(X_train[self.num_features].values)

        if len(self.cat_features) > 0:
            self.cat_imputer.fit(X_train[self.cat_features].values)
            self.cat_transformer = Pipeline(steps=[
                ('imputer', self.cat_imputer),
            ])

        self.label_enc.fit(y_train.values.ravel())

    def transform(self, X=None, y=None):
        if X is not None:
            X_after = []
            if len(self.num_features) > 0:
                X_arr = X[self.num_features].values
                if len(X_arr.shape) == 1:
                    X_arr = X_arr.reshape(1, -1)
                X_num = self.num_transformer.transform(X_arr)
                X_after.append(X_num)

            if len(self.cat_features) > 0:
                X_arr = X[self.cat_features].values
                if len(X_arr.shape) == 1:
                    X_arr = X_arr.reshape(1, -1)
                X_cat = self.cat_transformer.transform(X_arr)
                X_after.append(X_cat)

            X = np.hstack(X_after)
            X = X[:, np.argsort(self.features_index)]

        if y is not None:
            y = self.label_enc.transform(y.values.ravel())

        if X is None:
            return y
        elif y is None:
            return X
        else:
            return X, y

    def candidate(self, X):
        col_dom = {}
        for i in range(len(X[0])):
            for j in range(len(X)):
                if i in col_dom.keys():
                    col_dom[i].add(X[j][i])
                else:
                    col_dom[i] = set()
                    col_dom[i].add(X[j][i])

        for key in col_dom.keys():
            col_dom[key] = list(col_dom[key])

        return col_dom


def nclean_preprocess(data_dir, dataset, mv_prob=0.002, attack="random", percent=0.2):
    # load raw data and data info
    data_path = os.path.join(data_dir, dataset + '.csv')
    data = pd.read_csv(data_path)

    # dataset: bodyPerformance, company_bankruptcy_prediction, creditcard, employee,
    # fetal_health, fitness_class, heart, mushrooms, star_classification, winequalityN
    # split feature and label
    dataset_to_label_column_dict = {
        "bodyPerformance": "class",
        "company_bankruptcy_prediction": "Bankrupt",
        "creditcard": "Class",
        "employee": "LeaveOrNot",
        "fetal_health": "fetal_health",
        "fitness_class": "attended",
        "heart": "HeartDisease",
        "mushrooms": "class",
        "star_classification": "class",
        "winequalityN": "type",
    }
    label_column = dataset_to_label_column_dict[dataset]
    feature_column = [c for c in data.columns if c != label_column and "id" not in c.lower()]
    X_full = data[feature_column]
    y_full = data[[label_column]]

    np.random.seed(1)
    N = X_full.shape[0]
    idx = np.random.permutation(N)

    test_size = int(N * 0.2)

    # split train and test
    idx_test = idx[:test_size]
    idx_train = idx[test_size: N]

    # split X and y
    X_train = X_full.iloc[idx_train].reset_index(drop=True)
    y_train = y_full.iloc[idx_train].reset_index(drop=True)
    X_test = X_full.iloc[idx_test].reset_index(drop=True)
    y_test = y_full.iloc[idx_test].reset_index(drop=True)

    full_size = X_full.size
    train_size = X_train.size
    test_size = X_test.size

    # print(X_train)
    # print(X_train.columns)
    # print(size)

    X_train_mv = deepcopy(X_train)
    if attack == "random":
        # get missing prob matrix
        m_prob_matrix = np.ones(X_train_mv.shape) * mv_prob
        # inject missing values
        mask = np.random.rand(*X_train_mv.shape) <= m_prob_matrix
    elif attack == "feature":
        # get missing prob matrix
        m_prob_matrix = np.ones(X_train_mv.shape) * mv_prob
        random_list = np.random.choice(len(X_train_mv.columns), int(percent * len(X_train_mv.columns)), replace=False)
        # inject missing values
        mask = np.random.rand(*X_train_mv.shape) <= m_prob_matrix
        for col in range(len(X_train_mv.columns)):
            if col not in random_list:
                mask[:, col] = False
    else:
        raise ValueError("This attack is not implemented")

    # # avoid injecting in all columns for one row
    # for i in range(len(mask)):
    #     if mask[i].all():
    #         print("Bad Luck")
    #         non_mv = int(mask.shape[1] * (1 - mv_prob))
    #         non_mv_indices = np.random.choice(mask.shape[1], size=non_mv, replace=False)
    #         mask[i, non_mv_indices] = False

    X_train_mv[mask] = np.nan
    ind_mv = pd.DataFrame(mask, columns=X_train_mv.columns)
    # data["X_train_dirty"] = X_train_mv
    # data["indicator"] = ind_mv

    data_dict = {
        "X_train_clean": X_train, "y_train": y_train,
        "X_train_dirty": X_train_mv, "indicator": ind_mv,
        "X_full": X_full, "y_full": y_full,
        "X_test": X_test, "y_test": y_test,
        "full_size": full_size,
        "train_size": train_size,
        "test_size": test_size
    }

    return data_dict


def preprocess(data):
    X_full, y_full = data["X_full"], data["y_full"]
    X_train_dirty = data["X_train_dirty"]
    X_train_clean = data["X_train_clean"]
    y_train = data["y_train"]
    indicator = data["indicator"]
    X_test, y_test = data["X_test"], data["y_test"]
    full_size = data["full_size"]
    train_size = data["train_size"]
    test_size = data["test_size"]

    # preprocess data
    preprocessor = Preprocessor()
    preprocessor.fit(X_train_dirty, y_train)

    X_test, y_test = preprocessor.transform(X_test, y_test)
    X_train_clean, y_train = preprocessor.transform(X_train_clean, y_train)
    # updates here
    col_dom = preprocessor.candidate(X_train_clean)
    X_train_dirty = preprocessor.transform(X_train_dirty)

    y_train_set = set()
    for i in range(len(y_train)):
        y_train_set.add(y_train[i])
    y_train_set = list(y_train_set)

    data_after = {}
    data_after["X_train_mv"] = X_train_dirty
    data_after["X_train_clean"] = X_train_clean
    data_after["y_train"] = y_train
    data_after["y_train_set"] = y_train_set
    data_after["X_test"] = X_test
    data_after["y_test"] = y_test
    data_after["indicator"] = indicator
    data_after["col_dom"] = col_dom
    data_after["full_size"] = full_size
    data_after["train_size"] = train_size
    data_after["test_size"] = test_size

    return data_after

