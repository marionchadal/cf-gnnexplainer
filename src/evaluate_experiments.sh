#!/bin/bash

python evaluate.py --path=../results/107/SGD/107_cf_examples_lr0.01_beta0.5_mom0.9_epochs500_seed42
python evaluate.py --path=../results/107/SGD/107_cf_examples_lr0.1_beta0.5_mom0.9_epochs500_seed42
python evaluate.py --path=../results/107/SGD/107_cf_examples_lr0.01_beta0.1_mom0.9_epochs500_seed42
python evaluate.py --path=../results/107/SGD/107_cf_examples_lr0.1_beta0.1_mom0.9_epochs500_seed42
python evaluate.py --path=../results/107/SGD/107_cf_examples_lr0.01_beta0.1_mom0.1_epochs500_seed42
python evaluate.py --path=../results/107/SGD/107_cf_examples_lr0.1_beta0.1_mom0.1_epochs500_seed42
python evaluate.py --path=../results/107/SGD/107_cf_examples_lr0.01_beta0.5_mom0.1_epochs500_seed42
python evaluate.py --path=../results/107/SGD/107_cf_examples_lr0.1_beta0.5_mom0.1_epochs500_seed42