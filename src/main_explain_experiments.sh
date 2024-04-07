#!/bin/bash

python main_explain.py --dataset=circle1_degree_centrality_GT --lr=0.01 --beta=0.5 --n_momentum=0.9 --optimizer=SGD
python main_explain.py --dataset=circle1_degree_centrality_GT --lr=0.1 --beta=0.5 --n_momentum=0.9 --optimizer=SGD
python main_explain.py --dataset=circle1_degree_centrality_GT --lr=0.01 --beta=0.1 --n_momentum=0.9 --optimizer=SGD
python main_explain.py --dataset=circle1_degree_centrality_GT --lr=0.1 --beta=0.1 --n_momentum=0.9 --optimizer=SGD
python main_explain.py --dataset=circle1_degree_centrality_GT --lr=0.01 --beta=0.1 --n_momentum=0.1 --optimizer=SGD
python main_explain.py --dataset=circle1_degree_centrality_GT --lr=0.1 --beta=0.1 --n_momentum=0.1 --optimizer=SGD
python main_explain.py --dataset=circle1_degree_centrality_GT --lr=0.01 --beta=0.5 --n_momentum=0.1 --optimizer=SGD
python main_explain.py --dataset=circle1_degree_centrality_GT --lr=0.01 --beta=0.5 --n_momentum=0.1 --optimizer=SGD






