import os
import sys
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
from single_poisoning.alg import random_poisoning, smarter_random_poisoning, greedy_poisoning


def init():
    logging.basicConfig(level=logging.DEBUG)


def callback(result):
    print("Callback:", result)


def error_callback(error):
    print("Error:", error)


def single_poisoning_process(num_points, mv_prob, dataset, data_dir):
    data = nclean_preprocess(data_dir, dataset, mv_prob, attack="random", percent=0.2)
    data = preprocess(data)
    print("Preprocess Finished")

    train_size = data["train_size"]
    test_list = [i for i in range(len(data["X_test"]))]
    sample_test = random.sample(test_list, num_points)
    X_test_list = data["X_test"][sample_test]

    assert len(X_test_list) == 1
    X_test = X_test_list[0]
    filename = "../result/single_poisoning_" + str(dataset) \
               + '_' + str(num_points) \
               + '_' + str(mv_prob) \
               + '_' + re.sub(r'[^0-9]', '', str(datetime.datetime.now())) + '.txt'

    # random poisoning
    start_time = time.time()
    step = random_poisoning(data, X_test)
    end_time = time.time()
    run_time = end_time - start_time
    print(run_time, 's')
    with open(filename, 'a') as file:
        file.write("========== Random Poisoning (RP) ==========" + "\n")
        file.write("steps: " + str(step) + '\n')
        file.write("poisoning_rate: " + str(step / train_size) + '\n')
        file.write("total_run_time: " + str(run_time) + '\n')

    # random poisoning plus
    start_time = time.time()
    step = smarter_random_poisoning(data, X_test)
    end_time = time.time()
    run_time = end_time - start_time
    print(run_time, 's')
    with open(filename, 'a') as file:
        file.write("========== Smarter Random Algorithm (SR) ==========" + "\n")
        file.write("steps: " + str(step) + '\n')
        file.write("poisoning_rate: " + str(step / train_size) + '\n')
        file.write("total_run_time: " + str(run_time) + '\n')

    # optimal poisoning
    start_time = time.time()
    step = greedy_poisoning(data, X_test)
    end_time = time.time()
    run_time = end_time - start_time
    print(run_time, 's')
    with open(filename, 'a') as file:
        file.write("========== Greedy Search Poisoning (GS) ==========" + "\n")
        file.write("steps: " + str(step) + '\n')
        file.write("poisoning_rate: " + str(step / train_size) + '\n')
        file.write("total_run_time: " + str(run_time) + '\n')


# dataset: bodyPerformance, company_bankruptcy_prediction, creditcard, employee,
# fetal_health, fitness_class, heart, mushrooms, star_classification, winequalityN
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', default=["employee"], nargs='+')
    # parser.add_argument('--data_dir', default="../datasets")
    # parser.add_argument('--mv_prob', default=0.002)
    # parser.add_argument('--num_points', default=1)
    # parser.add_argument('--attack', default="random")
    # parser.add_argument('--percent', default=0.2)
    # args = parser.parse_args()

    dataset_list = ['bodyPerformance', 'company_bankruptcy_prediction', 'creditcard',
                    'employee', 'fetal_health', 'fitness_class', 'heart',
                    'mushrooms', 'star_classification', 'winequalityN']
    num_points_list = [1]
    mv_prob_list = [0.20]
    process_num = min(len(dataset_list) * len(mv_prob_list) * len(num_points_list), os.cpu_count())
    pool = multiprocessing.Pool(processes=process_num, initializer=init)

    results = []
    data_dir = "../datasets"
    for num_points in num_points_list:
        for mv_prob in mv_prob_list:
            for dataset in dataset_list:
                results.append(
                    pool.apply_async(
                        single_poisoning_process,
                        (num_points, mv_prob, dataset, data_dir),
                        callback=callback, error_callback=error_callback
                    )
                )

    pool.close()
    pool.join()







