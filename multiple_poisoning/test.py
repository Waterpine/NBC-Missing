import os
import sys
import argparse
import time
import re
import random
import datetime

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir)
    )
)

from tools.preprocess import preprocess, nclean_preprocess
from multiple_poisoning.alg import multiple_random_poisoning, multiple_smarter_random_poisoning, \
    multiple_greedy_poisoning


# dataset: bodyPerformance, company_bankruptcy_prediction, creditcard, employee,
# fetal_health, fitness_class, heart, mushrooms, star_classification, winequalityN
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=["fetal_health"], nargs='+')
    parser.add_argument('--data_dir', default="../datasets")
    parser.add_argument('--mv_prob', default=0.002)
    parser.add_argument('--num_points', default=2)
    parser.add_argument('--attack', default="random")
    parser.add_argument('--percent', default=0.2)
    args = parser.parse_args()

    num_points = int(args.num_points)
    mv_prob = float(args.mv_prob)
    dataset = args.dataset[0]
    data = nclean_preprocess(args.data_dir, dataset, mv_prob, args.attack, args.percent)
    data = preprocess(data)
    print("Preprocess Finished")

    train_size = data["train_size"]
    test_list = [i for i in range(len(data["X_test"]))]
    sample_test = random.sample(test_list, num_points)
    X_test_list = data["X_test"][sample_test]

    assert len(X_test_list) > 1
    filename = "../result/multiple_poisoning_" + str(dataset) \
               + '_' + str(args.num_points) \
               + '_' + str(args.mv_prob) \
               + '_' + re.sub(r'[^0-9]', '', str(datetime.datetime.now())) + '.txt'

    # # random poisoning
    # start_time = time.time()
    # step = multiple_random_poisoning(data, X_test_list)
    # end_time = time.time()
    # run_time = end_time - start_time
    # print(run_time, 's')
    # with open(filename, 'a') as file:
    #     file.write("========== Multiple Random Poisoning (RP) ==========" + "\n")
    #     file.write("steps: " + str(step) + '\n')
    #     file.write("poisoning_rate: " + str(step / train_size) + '\n')
    #     file.write("total_run_time: " + str(run_time) + '\n')
    #
    # random poisoning plus
    start_time = time.time()
    step = multiple_smarter_random_poisoning(data, X_test_list)
    end_time = time.time()
    run_time = end_time - start_time
    print("steps: ", step)
    print("poisoning_rate: ", step / train_size)
    print("total_run_time: ", run_time, 's')
    with open(filename, 'a') as file:
        file.write("========== Multiple Smarter Random Algorithm (SR) ==========" + "\n")
        file.write("steps: " + str(step) + '\n')
        file.write("poisoning_rate: " + str(step / train_size) + '\n')
        file.write("total_run_time: " + str(run_time) + '\n')

    # multiple optimal poisoning
    start_time = time.time()
    step = multiple_greedy_poisoning(data, X_test_list)
    end_time = time.time()
    run_time = end_time - start_time
    print("steps: ", step)
    print("poisoning_rate: ", step / train_size)
    print("total_run_time: ", run_time, 's')
    with open(filename, 'a') as file:
        file.write("========== Multiple Optimal Poisoning (OP) ==========" + "\n")
        file.write("steps: " + str(step) + '\n')
        file.write("poisoning_rate: " + str(step / train_size) + '\n')
        file.write("total_run_time: " + str(run_time) + '\n')






