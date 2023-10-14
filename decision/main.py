import os
import sys
import numpy as np
import argparse
import time
import re
import random
import datetime
import multiprocessing
import logging

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir)
    )
)

from tools.preprocess import preprocess, nclean_preprocess
from decision.alg import approximate_decision, iterative_algorithm, iterative_algorithm_with_index


def init():
    logging.basicConfig(level=logging.DEBUG)


def callback(result):
    print("Callback:", result)


def error_callback(error):
    print("Error:", error)


def decision_process(num_points, mv_prob, dataset, data_dir):
    data = nclean_preprocess(data_dir, dataset, mv_prob, attack="random", percent=0.2)
    data = preprocess(data)
    print("Preprocess Finished")

    indicator = data["indicator"].values
    print(np.sum(indicator, axis=0))
    print(indicator.shape)

    test_list = [i for i in range(len(data["X_test"]))]
    sample_test = random.sample(test_list, num_points)
    X_test_list = data["X_test"][sample_test]

    filename = "../result/decision_" + str(dataset) \
               + '_' + str(num_points) \
               + '_' + str(mv_prob) \
               + '_' + "random" \
               + '_' + str(0.2) \
               + '_' + re.sub(r'[^0-9]', '', str(datetime.datetime.now())) + '.txt'

    # # Approximate Decision
    # start_time = time.time()
    # approximate_decision(data, X_test_list)
    # end_time = time.time()
    # run_time = end_time - start_time
    # print(run_time, 's')
    # with open(filename, 'a') as file:
    #     file.write("========== Approximate Decision (AD) ==========" + "\n")
    #     file.write("total_run_time: " + str(run_time) + '\n')
    #     file.write("\n")

    # Iterative Algorithm
    start_time = time.time()
    iterative_algorithm(data, X_test_list)
    end_time = time.time()
    run_time = end_time - start_time
    print(run_time, 's')
    with open(filename, 'a') as file:
        file.write("========== Iterative Algorithm (Iterate) ==========" + "\n")
        file.write("total_run_time: " + str(run_time) + '\n')
        file.write("\n")

    # Iterative Algorithm with Index
    start_time = time.time()
    query_time = iterative_algorithm_with_index(data, X_test_list)
    end_time = time.time()
    run_time = end_time - start_time
    print(run_time, 's')
    with open(filename, 'a') as file:
        file.write("========== Iterative Algorithm with Index (Iterate+Index) ==========" + "\n")
        file.write("total_run_time: " + str(run_time) + '\n')
        file.write("query_time: " + str(query_time) + '\n')
        file.write("index_time: " + str(run_time - query_time) + '\n')
        file.write("\n")


# dataset: bodyPerformance, company_bankruptcy_prediction, creditcard, employee,
# fetal_health, fitness_class, heart, mushrooms, star_classification, winequalityN
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', default=["company_bankruptcy_prediction"], nargs='+')
    # parser.add_argument('--data_dir', default="../datasets")
    # parser.add_argument('--mv_prob', default=0.40)
    # parser.add_argument('--num_points', default=2)
    # parser.add_argument('--attack', default="random")
    # parser.add_argument('--percent', default=0.2)
    # args = parser.parse_args()

    # dataset_list = ['bodyPerformance', 'company_bankruptcy_prediction', 'creditcard', 'employee',
    #                 'fetal_health', 'fitness_class', 'heart', 'mushrooms',
    #                 'star_classification', 'winequalityN']
    dataset_list = ['creditcard']
    num_points_list = [16]
    mv_prob_list = [0.20, 0.40, 0.60, 0.80]
    # num_points_list = [1, 2, 4, 8, 16, 32]
    # mv_prob_list = [0.20, 0.40, 0.60, 0.80]
    process_num = min(len(dataset_list) * len(mv_prob_list) * len(num_points_list), os.cpu_count())
    pool = multiprocessing.Pool(processes=process_num, initializer=init)

    results = []
    data_dir = "../datasets"
    for num_points in num_points_list:
        for mv_prob in mv_prob_list:
            for dataset in dataset_list:
                results.append(
                    pool.apply_async(
                        decision_process,
                        (num_points, mv_prob, dataset, data_dir),
                        callback=callback, error_callback=error_callback
                    )
                )

    pool.close()
    pool.join()

