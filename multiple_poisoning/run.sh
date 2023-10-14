#!/bin/bash
for k in 1 2 3 4 5
do
  for dataset in 'bodyPerformance' 'company_bankruptcy_prediction' 'creditcard' 'employee' 'fetal_health' 'fitness_class' 'heart' 'mushrooms' 'star_classification' 'winequalityN'
  do
    for numpoints in 2 4 8 16
    do
      python3 test.py --dataset=$dataset --num_points=$numpoints
    done
  done
done