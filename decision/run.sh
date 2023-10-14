#!/bin/bash
for k in 1 2 3 4 5
do
  for dataset in 'bodyPerformance' 'company_bankruptcy_prediction' 'creditcard' 'employee' 'fetal_health' 'fitness_class' 'heart' 'mushrooms' 'star_classification' 'winequalityN'
  do
    for numpoints in 1 2 4 8 16 32
    do
      for mvprob in 0.02 0.04 0.06 0.08 0.10 0.20 0.40 0.60 0.80
      do
        python3 test.py --dataset=$dataset --num_points=$numpoints --mv_prob=$mvprob
      done
    done
  done
done

# dataset: bodyPerformance, company_bankruptcy_prediction, creditcard, employee,
# fetal_health, fitness_class, heart, mushrooms, star_classification, winequalityN