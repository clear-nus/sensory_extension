#!/bin/sh
# python evaluate_kernel_features.py kernel_regression_rodTap_50_4000_all_mlp
# python evaluate_kernel_features.py kernel_regressionSingleTax_rodTap_30_4000_neuhalf_svmrbf

# python evaluate_kernel_features.py kernel_regression_rodTap_20_1000_all_svrlinear
# python evaluate_kernel_features.py kernel_classification_handover_1_4000_neuhalf_svmlinear

python evaluate_kernel_features.py kernel_regression_rodTap_30_4000_neuhalf_rnn
python evaluate_kernel_features.py kernel_classification_handover_2_4000_neuhalf_rnn