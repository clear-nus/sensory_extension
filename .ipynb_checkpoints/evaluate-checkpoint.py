from functools import partial

from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC

from skorch import NeuralNetRegressor, NeuralNetClassifier

import torch
import torch.nn as nn

import numpy as np

def evaluate(experiment_name, device='cuda'):
    
    experiments = {
        
        'tool20_biotac_baseline_svmlinear'                 : partial(_evaluate_tool_svm, 20, 'biotac', False, 'linear'),
        'tool20_biotac_baseline_svmrbf'                    : partial(_evaluate_tool_svm, 20, 'biotac', False, 'rbf'),
        'tool20_biotac_baseline_mlp'                       : partial(_evaluate_tool_mlp, 20, 'biotac', False),
        'tool20_biotac_baseline_rnn'                       : partial(_evaluate_tool_rnn, 20, 'biotac', device),
        'tool20_biotac_fft_svmlinear'                      : partial(_evaluate_tool_svm, 20, 'biotac', True, 'linear'),
        'tool20_biotac_fft_svmrbf'                         : partial(_evaluate_tool_svm, 20, 'biotac', True, 'rbf'),
        'tool20_biotac_fft_mlp'                            : partial(_evaluate_tool_mlp, 20, 'biotac', True),
        
        'tool20_neutouch_baseline_svmlinear'               : partial(_evaluate_tool_svm, 20, 'neutouch', False, 'linear'),
        'tool20_neutouch_baseline_svmrbf'                  : partial(_evaluate_tool_svm, 20, 'neutouch', False, 'rbf'),
        'tool20_neutouch_baseline_mlp'                     : partial(_evaluate_tool_mlp, 20, 'neutouch', False),
        'tool20_neutouch_baseline_rnn'                     : partial(_evaluate_tool_rnn, 20, 'neutouch', device),
        'tool20_neutouch_fft_svmlinear'                    : partial(_evaluate_tool_svm, 20, 'neutouch', True, 'linear'),
        'tool20_neutouch_fft_svmrbf'                       : partial(_evaluate_tool_svm, 20, 'neutouch', True, 'rbf'),
        'tool20_neutouch_fft_mlp'                          : partial(_evaluate_tool_mlp, 20, 'neutouch', True),
        
        'tool20_neuhalf_baseline_svmlinear'                : partial(_evaluate_tool_svm, 20, 'neuhalf', False, 'linear'),
        'tool20_neuhalf_baseline_svmrbf'                   : partial(_evaluate_tool_svm, 20, 'neuhalf', False, 'rbf'),
        'tool20_neuhalf_baseline_mlp'                      : partial(_evaluate_tool_mlp, 20, 'neuhalf', False),
        'tool20_neuhalf_baseline_rnn'                      : partial(_evaluate_tool_rnn, 20, 'neuhalf', device),
        'tool20_neuhalf_fft_svmlinear'                     : partial(_evaluate_tool_svm, 20, 'neuhalf', True, 'linear'),
        'tool20_neuhalf_fft_svmrbf'                        : partial(_evaluate_tool_svm, 20, 'neuhalf', True, 'rbf'),
        'tool20_neuhalf_fft_mlp'                           : partial(_evaluate_tool_mlp, 20, 'neuhalf', True),
        
        'tool30_biotac_baseline_svmlinear'                 : partial(_evaluate_tool_svm, 30, 'biotac', False, 'linear'),
        'tool30_biotac_baseline_svmrbf'                    : partial(_evaluate_tool_svm, 30, 'biotac', False, 'rbf'),
        'tool30_biotac_baseline_mlp'                       : partial(_evaluate_tool_mlp, 30, 'biotac', False),
        'tool30_biotac_baseline_rnn'                       : partial(_evaluate_tool_rnn, 30, 'biotac', device),
        'tool30_biotac_fft_svmlinear'                      : partial(_evaluate_tool_svm, 30, 'biotac', True, 'linear'),
        'tool30_biotac_fft_svmrbf'                         : partial(_evaluate_tool_svm, 30, 'biotac', True, 'rbf'),
        'tool30_biotac_fft_mlp'                            : partial(_evaluate_tool_mlp, 30, 'biotac', True),
        
        'tool30_neutouch_baseline_svmlinear'               : partial(_evaluate_tool_svm, 30, 'neutouch', False, 'linear'),
        'tool30_neutouch_baseline_svmrbf'                  : partial(_evaluate_tool_svm, 30, 'neutouch', False, 'rbf'),
        'tool30_neutouch_baseline_mlp'                     : partial(_evaluate_tool_mlp, 30, 'neutouch', False),
        'tool30_neutouch_baseline_rnn'                     : partial(_evaluate_tool_rnn, 30, 'neutouch', device),
        'tool30_neutouch_fft_svmlinear'                    : partial(_evaluate_tool_svm, 30, 'neutouch', True, 'linear'),
        'tool30_neutouch_fft_svmrbf'                       : partial(_evaluate_tool_svm, 30, 'neutouch', True, 'rbf'),
        'tool30_neutouch_fft_mlp'                          : partial(_evaluate_tool_mlp, 30, 'neutouch', True),
        
        'tool30_neuhalf_baseline_svmlinear'                : partial(_evaluate_tool_svm, 30, 'neuhalf', False, 'linear'),
        'tool30_neuhalf_baseline_svmrbf'                   : partial(_evaluate_tool_svm, 30, 'neuhalf', False, 'rbf'),
        'tool30_neuhalf_baseline_mlp'                      : partial(_evaluate_tool_mlp, 30, 'neuhalf', False),
        'tool30_neuhalf_baseline_rnn'                      : partial(_evaluate_tool_rnn, 30, 'neuhalf', device),
        'tool30_neuhalf_fft_svmlinear'                     : partial(_evaluate_tool_svm, 30, 'neuhalf', True, 'linear'),
        'tool30_neuhalf_fft_svmrbf'                        : partial(_evaluate_tool_svm, 30, 'neuhalf', True, 'rbf'),
        'tool30_neuhalf_fft_mlp'                           : partial(_evaluate_tool_mlp, 30, 'neuhalf', True),
        
        'tool50_biotac_baseline_svmlinear'                 : partial(_evaluate_tool_svm, 50, 'biotac', False, 'linear'),
        'tool50_biotac_baseline_svmrbf'                    : partial(_evaluate_tool_svm, 50, 'biotac', False, 'rbf'),
        'tool50_biotac_baseline_mlp'                       : partial(_evaluate_tool_mlp, 50, 'biotac', False),
        'tool50_biotac_baseline_rnn'                       : partial(_evaluate_tool_rnn, 50, 'biotac', device),
        'tool50_biotac_fft_svmlinear'                      : partial(_evaluate_tool_svm, 50, 'biotac', True, 'linear'),
        'tool50_biotac_fft_svmrbf'                         : partial(_evaluate_tool_svm, 50, 'biotac', True, 'rbf'),
        'tool50_biotac_fft_mlp'                            : partial(_evaluate_tool_mlp, 50, 'biotac', True),
        
        'tool50_neutouch_baseline_svmlinear'               : partial(_evaluate_tool_svm, 50, 'neutouch', False, 'linear'),
        'tool50_neutouch_baseline_svmrbf'                  : partial(_evaluate_tool_svm, 50, 'neutouch', False, 'rbf'),
        'tool50_neutouch_baseline_mlp'                     : partial(_evaluate_tool_mlp, 50, 'neutouch', False),
        'tool50_neutouch_baseline_rnn'                     : partial(_evaluate_tool_rnn, 50, 'neutouch', device),
        'tool50_neutouch_fft_svmlinear'                    : partial(_evaluate_tool_svm, 50, 'neutouch', True, 'linear'),
        'tool50_neutouch_fft_svmrbf'                       : partial(_evaluate_tool_svm, 50, 'neutouch', True, 'rbf'),
        'tool50_neutouch_fft_mlp'                          : partial(_evaluate_tool_mlp, 50, 'neutouch', True),
        
        'tool50_neuhalf_baseline_svmlinear'                : partial(_evaluate_tool_svm, 50, 'neuhalf', False, 'linear'),
        'tool50_neuhalf_baseline_svmrbf'                   : partial(_evaluate_tool_svm, 50, 'neuhalf', False, 'rbf'),
        'tool50_neuhalf_baseline_mlp'                      : partial(_evaluate_tool_mlp, 50, 'neuhalf', False),
        'tool50_neuhalf_baseline_rnn'                      : partial(_evaluate_tool_rnn, 50, 'neuhalf', device),
        'tool50_neuhalf_fft_svmlinear'                     : partial(_evaluate_tool_svm, 50, 'neuhalf', True, 'linear'),
        'tool50_neuhalf_fft_svmrbf'                        : partial(_evaluate_tool_svm, 50, 'neuhalf', True, 'rbf'),
        'tool50_neuhalf_fft_mlp'                           : partial(_evaluate_tool_mlp, 50, 'neuhalf', True),
        
        'tool20_biotac_autoencoder_svmlinear'              : partial(_evaluate_tool_biotac_aesvm, 20, 'linear'),
        'tool20_biotac_autoencoder_svmrbf'                 : partial(_evaluate_tool_biotac_aesvm, 20, 'rbf'),
        'tool20_biotac_autoencoder_mlp'                    : partial(_evaluate_tool_biotac_aemlp, 20),
        
        'tool20_neutouch_autoencoder_svmlinear'            : partial(_evaluate_tool_neutouch_aesvm, 20, 'linear'),
        'tool20_neutouch_autoencoder_svmrbf'               : partial(_evaluate_tool_neutouch_aesvm, 20, 'rbf'),
        'tool20_neutouch_autoencoder_mlp'                  : partial(_evaluate_tool_neutouch_aemlp, 20),
        
        'tool30_biotac_autoencoder_svmlinear'              : partial(_evaluate_tool_biotac_aesvm, 30, 'linear'),
        'tool30_biotac_autoencoder_svmrbf'                 : partial(_evaluate_tool_biotac_aesvm, 30, 'rbf'),
        'tool30_biotac_autoencoder_mlp'                    : partial(_evaluate_tool_biotac_aemlp, 30),
        
        'tool30_neutouch_autoencoder_svmlinear'            : partial(_evaluate_tool_neutouch_aesvm, 30, 'linear'),
        'tool30_neutouch_autoencoder_svmrbf'               : partial(_evaluate_tool_neutouch_aesvm, 30, 'rbf'),
        'tool30_neutouch_autoencoder_mlp'                  : partial(_evaluate_tool_neutouch_aemlp, 30),
        
        'tool50_biotac_autoencoder_svmlinear'              : partial(_evaluate_tool_biotac_aesvm, 50, 'linear'),
        'tool50_biotac_autoencoder_svmrbf'                 : partial(_evaluate_tool_biotac_aesvm, 50, 'rbf'),
        'tool50_biotac_autoencoder_mlp'                    : partial(_evaluate_tool_biotac_aemlp, 50),
        
        'tool50_neutouch_autoencoder_svmlinear'            : partial(_evaluate_tool_neutouch_aesvm, 50, 'linear'),
        'tool50_neutouch_autoencoder_svmrbf'               : partial(_evaluate_tool_neutouch_aesvm, 50, 'rbf'),
        'tool50_neutouch_autoencoder_mlp'                  : partial(_evaluate_tool_neutouch_aemlp, 50),
        
        'tool20_neusingle_baseline_svmlinear'              : partial(_evaluate_tool_neusingle_svm, 20, 'linear', False),
        'tool20_neusingle_baseline_svmrbf'                 : partial(_evaluate_tool_neusingle_svm, 20, 'rbf', False),
        'tool20_neusingle_fft_svmlinear'                   : partial(_evaluate_tool_neusingle_svm, 20, 'linear', True),
        'tool20_neusingle_fft_svmrbf'                      : partial(_evaluate_tool_neusingle_svm, 20, 'rbf', True),
        
        'tool30_neusingle_baseline_svmlinear'              : partial(_evaluate_tool_neusingle_svm, 30, 'linear', False),
        'tool30_neusingle_baseline_svmrbf'                 : partial(_evaluate_tool_neusingle_svm, 30, 'rbf', False),
        'tool30_neusingle_fft_svmlinear'                   : partial(_evaluate_tool_neusingle_svm, 30, 'linear', True),
        'tool30_neusingle_fft_svmrbf'                      : partial(_evaluate_tool_neusingle_svm, 30, 'rbf', True),
        
        'tool50_neusingle_baseline_svmlinear'              : partial(_evaluate_tool_neusingle_svm, 50, 'linear', False),
        'tool50_neusingle_baseline_svmrbf'                 : partial(_evaluate_tool_neusingle_svm, 50, 'rbf', False),
        'tool50_neusingle_fft_svmlinear'                   : partial(_evaluate_tool_neusingle_svm, 50, 'linear', True),
        'tool50_neusingle_fft_svmrbf'                      : partial(_evaluate_tool_neusingle_svm, 50, 'rbf', True),
        
        'handoverrod_biotac_baseline_svmlinear'            : partial(_evaluate_handover_svm, 'rod', 'biotac', False, 'linear'),
        'handoverrod_biotac_baseline_svmrbf'               : partial(_evaluate_handover_svm, 'rod', 'biotac', False, 'rbf'),
        'handoverrod_biotac_baseline_mlp'                  : partial(_evaluate_handover_mlp, 'rod', 'biotac', False),
        'handoverrod_biotac_baseline_rnn'                  : partial(_evaluate_handover_rnn, 'rod', 'biotac', device),
        'handoverrod_biotac_fft_svmlinear'                 : partial(_evaluate_handover_svm, 'rod', 'biotac', True, 'linear'),
        'handoverrod_biotac_fft_svmrbf'                    : partial(_evaluate_handover_svm, 'rod', 'biotac', True, 'rbf'),
        'handoverrod_biotac_fft_mlp'                       : partial(_evaluate_handover_mlp, 'rod', 'biotac', True),
        
        'handoverrod_neutouch_baseline_svmlinear'          : partial(_evaluate_handover_svm, 'rod', 'neutouch', False, 'linear'),
        'handoverrod_neutouch_baseline_svmrbf'             : partial(_evaluate_handover_svm, 'rod', 'neutouch', False, 'rbf'),
        'handoverrod_neutouch_baseline_mlp'                : partial(_evaluate_handover_mlp, 'rod', 'neutouch', False),
        'handoverrod_neutouch_baseline_rnn'                : partial(_evaluate_handover_rnn, 'rod', 'neutouch', device),
        'handoverrod_neutouch_fft_svmlinear'               : partial(_evaluate_handover_svm, 'rod', 'neutouch', True, 'linear'),
        'handoverrod_neutouch_fft_svmrbf'                  : partial(_evaluate_handover_svm, 'rod', 'neutouch', True, 'rbf'),
        'handoverrod_neutouch_fft_mlp'                     : partial(_evaluate_handover_mlp, 'rod', 'neutouch', True),
        
        'handoverrod_neuhalf_baseline_svmlinear'           : partial(_evaluate_handover_svm, 'rod', 'neuhalf', False, 'linear'),
        'handoverrod_neuhalf_baseline_svmrbf'              : partial(_evaluate_handover_svm, 'rod', 'neuhalf', False, 'rbf'),
        'handoverrod_neuhalf_baseline_mlp'                 : partial(_evaluate_handover_mlp, 'rod', 'neuhalf', False),
        'handoverrod_neuhalf_baseline_rnn'                 : partial(_evaluate_handover_rnn, 'rod', 'neuhalf', device),
        'handoverrod_neuhalf_fft_svmlinear'                : partial(_evaluate_handover_svm, 'rod', 'neuhalf', True, 'linear'),
        'handoverrod_neuhalf_fft_svmrbf'                   : partial(_evaluate_handover_svm, 'rod', 'neuhalf', True, 'rbf'),
        'handoverrod_neuhalf_fft_mlp'                      : partial(_evaluate_handover_mlp, 'rod', 'neuhalf', True),
        
        'handoverbox_biotac_baseline_svmlinear'            : partial(_evaluate_handover_svm, 'box', 'biotac', False, 'linear'),
        'handoverbox_biotac_baseline_svmrbf'               : partial(_evaluate_handover_svm, 'box', 'biotac', False, 'rbf'),
        'handoverbox_biotac_baseline_mlp'                  : partial(_evaluate_handover_mlp, 'box', 'biotac', False),
        'handoverbox_biotac_baseline_rnn'                  : partial(_evaluate_handover_rnn, 'box', 'biotac', device),
        'handoverbox_biotac_fft_svmlinear'                 : partial(_evaluate_handover_svm, 'box', 'biotac', True, 'linear'),
        'handoverbox_biotac_fft_svmrbf'                    : partial(_evaluate_handover_svm, 'box', 'biotac', True, 'rbf'),
        'handoverbox_biotac_fft_mlp'                       : partial(_evaluate_handover_mlp, 'box', 'biotac', True),
        
        'handoverbox_neutouch_baseline_svmlinear'          : partial(_evaluate_handover_svm, 'box', 'neutouch', False, 'linear'),
        'handoverbox_neutouch_baseline_svmrbf'             : partial(_evaluate_handover_svm, 'box', 'neutouch', False, 'rbf'),
        'handoverbox_neutouch_baseline_mlp'                : partial(_evaluate_handover_mlp, 'box', 'neutouch', False),
        'handoverbox_neutouch_baseline_rnn'                : partial(_evaluate_handover_rnn, 'box', 'neutouch', device),
        'handoverbox_neutouch_fft_svmlinear'               : partial(_evaluate_handover_svm, 'box', 'neutouch', True, 'linear'),
        'handoverbox_neutouch_fft_svmrbf'                  : partial(_evaluate_handover_svm, 'box', 'neutouch', True, 'rbf'),
        'handoverbox_neutouch_fft_mlp'                     : partial(_evaluate_handover_mlp, 'box', 'neutouch', True),
        
        'handoverbox_neuhalf_baseline_svmlinear'           : partial(_evaluate_handover_svm, 'box', 'neuhalf', False, 'linear'),
        'handoverbox_neuhalf_baseline_svmrbf'              : partial(_evaluate_handover_svm, 'box', 'neuhalf', False, 'rbf'),
        'handoverbox_neuhalf_baseline_mlp'                 : partial(_evaluate_handover_mlp, 'box', 'neuhalf', False),
        'handoverbox_neuhalf_baseline_rnn'                 : partial(_evaluate_handover_rnn, 'box', 'neuhalf', device),
        'handoverbox_neuhalf_fft_svmlinear'                : partial(_evaluate_handover_svm, 'box', 'neuhalf', True, 'linear'),
        'handoverbox_neuhalf_fft_svmrbf'                   : partial(_evaluate_handover_svm, 'box', 'neuhalf', True, 'rbf'),
        'handoverbox_neuhalf_fft_mlp'                      : partial(_evaluate_handover_mlp, 'box', 'neuhalf', True),
        
        'handoverplate_biotac_baseline_svmlinear'          : partial(_evaluate_handover_svm, 'plate', 'biotac', False, 'linear'),
        'handoverplate_biotac_baseline_svmrbf'             : partial(_evaluate_handover_svm, 'plate', 'biotac', False, 'rbf'),
        'handoverplate_biotac_baseline_mlp'                : partial(_evaluate_handover_mlp, 'plate', 'biotac', False),
        'handoverplate_biotac_baseline_rnn'                : partial(_evaluate_handover_rnn, 'plate', 'biotac', device),
        'handoverplate_biotac_fft_svmlinear'               : partial(_evaluate_handover_svm, 'plate', 'biotac', True, 'linear'),
        'handoverplate_biotac_fft_svmrbf'                  : partial(_evaluate_handover_svm, 'plate', 'biotac', True, 'rbf'),
        'handoverplate_biotac_fft_mlp'                     : partial(_evaluate_handover_mlp, 'plate', 'biotac', True),
        
        'handoverplate_neutouch_baseline_svmlinear'        : partial(_evaluate_handover_svm, 'plate', 'neutouch', False, 'linear'),
        'handoverplate_neutouch_baseline_svmrbf'           : partial(_evaluate_handover_svm, 'plate', 'neutouch', False, 'rbf'),
        'handoverplate_neutouch_baseline_mlp'              : partial(_evaluate_handover_mlp, 'plate', 'neutouch', False),
        'handoverplate_neutouch_baseline_rnn'              : partial(_evaluate_handover_rnn, 'plate', 'neutouch', device),
        'handoverplate_neutouch_fft_svmlinear'             : partial(_evaluate_handover_svm, 'plate', 'neutouch', True, 'linear'),
        'handoverplate_neutouch_fft_svmrbf'                : partial(_evaluate_handover_svm, 'plate', 'neutouch', True, 'rbf'),
        'handoverplate_neutouch_fft_mlp'                   : partial(_evaluate_handover_mlp, 'plate', 'neutouch', True),
        
        'handoverplate_neuhalf_baseline_svmlinear'         : partial(_evaluate_handover_svm, 'plate', 'neuhalf', False, 'linear'),
        'handoverplate_neuhalf_baseline_svmrbf'            : partial(_evaluate_handover_svm, 'plate', 'neuhalf', False, 'rbf'),
        'handoverplate_neuhalf_baseline_mlp'               : partial(_evaluate_handover_mlp, 'plate', 'neuhalf', False),
        'handoverplate_neuhalf_baseline_rnn'               : partial(_evaluate_handover_rnn, 'plate', 'neuhalf', device),
        'handoverplate_neuhalf_fft_svmlinear'              : partial(_evaluate_handover_svm, 'plate', 'neuhalf', True, 'linear'),
        'handoverplate_neuhalf_fft_svmrbf'                 : partial(_evaluate_handover_svm, 'plate', 'neuhalf', True, 'rbf'),
        'handoverplate_neuhalf_fft_mlp'                    : partial(_evaluate_handover_mlp, 'plate', 'neuhalf', True),
        
        'food_biotac_baseline_svmlinear'                   : partial(_evaluate_food_svm, 'biotac', False, 'linear'),
        'food_biotac_baseline_svmrbf'                      : partial(_evaluate_food_svm, 'biotac', False, 'rbf'),
        'food_biotac_baseline_mlp'                         : partial(_evaluate_food_mlp, 'biotac', False),
        'food_biotac_baseline_rnn'                         : partial(_evaluate_food_rnn, 'biotac', device),
        'food_biotac_fft_svmlinear'                        : partial(_evaluate_food_svm, 'biotac', True, 'linear'),
        'food_biotac_fft_svmrbf'                           : partial(_evaluate_food_svm, 'biotac', True, 'rbf'),
        'food_biotac_fft_mlp'                              : partial(_evaluate_food_mlp, 'biotac', True),
        
        'food_neutouch_baseline_svmlinear'                 : partial(_evaluate_food_svm, 'neutouch', False, 'linear'),
        'food_neutouch_baseline_svmrbf'                    : partial(_evaluate_food_svm, 'neutouch', False, 'rbf'),
        'food_neutouch_baseline_mlp'                       : partial(_evaluate_food_mlp, 'neutouch', False),
        'food_neutouch_baseline_rnn'                       : partial(_evaluate_food_rnn, 'neutouch', device),
        'food_neutouch_fft_svmlinear'                      : partial(_evaluate_food_svm, 'neutouch', True, 'linear'),
        'food_neutouch_fft_svmrbf'                         : partial(_evaluate_food_svm, 'neutouch', True, 'rbf'),
        'food_neutouch_fft_mlp'                            : partial(_evaluate_food_mlp, 'neutouch', True),
        
        'food_neuhalf_baseline_svmlinear'                  : partial(_evaluate_food_svm, 'neuhalf', False, 'linear'),
        'food_neuhalf_baseline_svmrbf'                     : partial(_evaluate_food_svm, 'neuhalf', False, 'rbf'),
        'food_neuhalf_baseline_mlp'                        : partial(_evaluate_food_mlp, 'neuhalf', False),
        'food_neuhalf_baseline_rnn'                        : partial(_evaluate_food_rnn, 'neuhalf', device),
        'food_neuhalf_fft_svmlinear'                       : partial(_evaluate_food_svm, 'neuhalf', True, 'linear'),
        'food_neuhalf_fft_svmrbf'                          : partial(_evaluate_food_svm, 'neuhalf', True, 'rbf'),
        'food_neuhalf_fft_mlp'                             : partial(_evaluate_food_mlp, 'neuhalf', True),
        
    }
    
    if experiment_name not in experiments: raise Exception('Experiment not found')
    
    test_loss_mean, test_loss_std = experiments[experiment_name]()
    
    print('Result for {:s}: {:0.4f} ± {:0.4f}'.format(experiment_name, test_loss_mean, test_loss_std))

    
class RNNModule(nn.Module):

    def __init__(self, input_dim, output_dim):
        
        super(RNNModule, self).__init__()
        
        self.rnn = nn.LSTM(input_dim, 16, batch_first=True)
        self.linear1 = nn.Linear(16, 8)
        self.linear2 = nn.Linear(8, output_dim)

    def forward(self, X):
        
        X, _ = self.rnn(X)
        X = torch.squeeze(X[:, -1, :])
        X = torch.relu(self.linear1(X))
        X = self.linear2(X)
        
        return X


#    ______          _             _                 
#   |  ____|        | |           | |                
#   | |____   ____ _| |_   _  __ _| |_ ___  _ __ ___ 
#   |  __\ \ / / _` | | | | |/ _` | __/ _ \| '__/ __|
#   | |___\ V / (_| | | |_| | (_| | || (_) | |  \__ \
#   |______\_/ \__,_|_|\__,_|\__,_|\__\___/|_|  |___/
#                                                    


def _create_evaluator(estimator, param_grid, scoring, cv=4, N=5, callback=None):
    
    gs_estimator = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=3, refit=True)
    
    def evaluate(X, y, verbose=True):
        
        test_losses = np.zeros(N)
        
        for n in range(N):
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=n)
            gs_estimator.fit(X_train, y_train)
            test_loss = -gs_estimator.score(X_test, y_test)
            test_losses[n] = test_loss
            
            if callback is not None: callback(gs_estimator, X_test, y_test)
            if verbose: print('Iteration {:d} | Test Loss = {:0.4f}'.format(n, test_loss))

        return np.mean(test_losses), np.std(test_losses)

    return evaluate


#    _______          _    ______                      _                      _       
#   |__   __|        | |  |  ____|                    (_)                    | |      
#      | | ___   ___ | |  | |__  __  ___ __   ___ _ __ _ _ __ ___   ___ _ __ | |_ ___ 
#      | |/ _ \ / _ \| |  |  __| \ \/ / '_ \ / _ \ '__| | '_ ` _ \ / _ \ '_ \| __/ __|
#      | | (_) | (_) | |  | |____ >  <| |_) |  __/ |  | | | | | | |  __/ | | | |_\__ \
#      |_|\___/ \___/|_|  |______/_/\_\ .__/ \___|_|  |_|_| |_| |_|\___|_| |_|\__|___/
#                                     | |                                             


def _load_tool(tool_length, signal_type, transformation):
    
    if signal_type == 'biotac':
        
        npzfile = np.load(f'preprocessed/biotac_tool_{tool_length}.npz')
        X = npzfile['signals'] / 1000
        y = npzfile['labels'] * 100
    
    if signal_type == 'neutouch':
        
        npzfile = np.load(f'preprocessed/neutouch_tool_{tool_length}.npz')
        X = npzfile['signals'] / 40
        y = npzfile['labels'] * 100
    
    if signal_type == 'neuhalf':
        
        npzfile = np.load(f'preprocessed/neutouch_tool_{tool_length}.npz')
        X = npzfile['signals'][:, 0:40, :] / 40
        y = npzfile['labels'] * 100
        
    if transformation == 'default':
        
        X = np.reshape(X, (X.shape[0], -1))
    
    if transformation == 'fft':
        
        X = np.abs(np.fft.fft(X)) / 10
        X = np.reshape(X, (X.shape[0], -1))
    
    if transformation == 'tensor' and signal_type == 'biotac':
    
        X = torch.Tensor(np.expand_dims(X, 2))
        y = torch.Tensor(np.reshape(y, (-1, 1)))
    
    if transformation == 'tensor' and signal_type == 'neutouch':
        
        X = torch.Tensor(np.swapaxes(X, 1, 2))
        y = torch.Tensor(np.reshape(y, (-1, 1)))
    
    return X, y


def _evaluate_tool_svm(tool_length, signal_type, perform_fft, kernel):

    X, y = _load_tool(tool_length, signal_type, 'fft' if perform_fft else 'default')
    
    param_grid = { 'C': [1, 3, 10, 30, 100] }
    
    estimator = SVR(kernel=kernel, max_iter=5000)
    evaluate = _create_evaluator(estimator, param_grid, 'neg_mean_absolute_error')
    
    return evaluate(X, y)


def _evaluate_tool_mlp(tool_length, signal_type, perform_fft):

    X, y = _load_tool(tool_length, signal_type, 'fft' if perform_fft else 'default')
    
    param_grid = {
        'learning_rate_init': [0.01, 0.03, 0.1, 0.3],
        'alpha': [0.0001, 0.001]
    }
    
    estimator = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=2000)
    evaluate = _create_evaluator(estimator, param_grid, 'neg_mean_absolute_error')
    
    return evaluate(X, y)


def _evaluate_tool_rnn(tool_length, signal_type, device):
    
    X, y = _load_tool(tool_length, signal_type, 'tensor')
    
    param_grid = { 'lr': [0.001, 0.003, 0.01] }
    
    estimator = NeuralNetRegressor(RNNModule,
                                   module__input_dim=X.shape[2],
                                   module__output_dim=1,
                                   iterator_train__shuffle=True,
                                   max_epochs=1000,
                                   train_split=False,
                                   device='cuda:0',
                                   verbose=1)
    
    evaluate = _create_evaluator(estimator,
                                 param_grid,
                                 'neg_mean_absolute_error',
                                 ShuffleSplit(n_splits=1, test_size=.25))
    
    return evaluate(X, y)


def _evaluate_tool_aesvm(tool_length, kernel):

    pass


def _evaluate_tool_aemlp(tool_length):

    pass


def _evaluate_tool_neusingle_svm(tool_length, kernel, perform_fft):

    X, y = _load_tool(tool_length, 'neutouch', 'fft' if perform_fft else 'default')
    X = np.reshape(X, (X.shape[0], 80, -1))

    best_taxel = 0
    best_test_loss_mean = float('inf')
    best_test_loss_std = float('inf')
    
    with open(f'results/neutouch_singletool_{tool_length}.csv', 'w') as file:
    
        for taxel in range(1, 81):
            
            param_grid = { 'C': [1, 3, 10, 30, 100] }
    
            estimator = SVR(kernel=kernel, max_iter=5000)
            evaluate = _create_evaluator(estimator, param_grid, 'neg_mean_absolute_error')
            test_loss_mean, test_loss_std = evaluate(X[:, taxel-1, :], y)
            file.write(f'{taxel},{test_loss_mean},{test_loss_std}\n')

            if test_loss_mean < best_test_loss_mean:

                best_taxel = taxel
                best_test_loss_mean = test_loss_mean
                best_test_loss_std = test_loss_std

            print('Result for taxel {:02d}: {:0.4f} ± {:0.4f}'.format(taxel, test_loss_mean, test_loss_std), flush=True)
    
    print(f'Best performing taxel is {best_taxel}')
    
    return best_test_loss_mean, best_test_loss_std


#    _    _                 _                        ______                      _                      _       
#   | |  | |               | |                      |  ____|                    (_)                    | |      
#   | |__| | __ _ _ __   __| | _____   _____ _ __   | |__  __  ___ __   ___ _ __ _ _ __ ___   ___ _ __ | |_ ___ 
#   |  __  |/ _` | '_ \ / _` |/ _ \ \ / / _ \ '__|  |  __| \ \/ / '_ \ / _ \ '__| | '_ ` _ \ / _ \ '_ \| __/ __|
#   | |  | | (_| | | | | (_| | (_) \ V /  __/ |     | |____ >  <| |_) |  __/ |  | | | | | | |  __/ | | | |_\__ \
#   |_|  |_|\__,_|_| |_|\__,_|\___/ \_/ \___|_|     |______/_/\_\ .__/ \___|_|  |_|_| |_| |_|\___|_| |_|\__|___/
#                                                               | |                                             


def _load_handover(item, signal_type, transformation):
    
    if signal_type == 'biotac':
        
        npzfile = np.load(f'preprocessed/biotac_handover_{item}.npz')
        X = npzfile['signals'] / 1000
        y = npzfile['labels'][:, 0].astype(np.compat.long)
    
    if signal_type == 'neutouch':
        
        npzfile = np.load(f'preprocessed/neutouch_handover_{item}.npz')
        X = npzfile['signals'] / 40
        y = npzfile['labels'][:, 0].astype(np.compat.long)
    
    if signal_type == 'neuhalf':
        
        npzfile = np.load(f'preprocessed/neutouch_handover_{item}.npz')
        X = npzfile['signals'][:, 0:40, :] / 40
        y = npzfile['labels'][:, 0].astype(np.compat.long)
        
    if transformation == 'default':
        
        X = np.reshape(X, (X.shape[0], -1))
    
    if transformation == 'fft':
        
        X = np.abs(np.fft.fft(X)) / 10
        X = np.reshape(X, (X.shape[0], -1))
    
    if transformation == 'tensor' and signal_type == 'biotac':
    
        X = torch.Tensor(np.swapaxes(X, 1, 2))
        y = torch.Tensor(np.reshape(y, (-1, 1)))
    
    if transformation == 'tensor' and signal_type == 'neutouch':
        
        torch.Tensor(np.expand_dims(X, 2))
        y = torch.Tensor(np.reshape(y, (-1, 1)))
    
    return X, y


def _evaluate_handover_svm(item, signal_type, perform_fft, kernel):

    X, y = _load_handover(item, signal_type, 'fft' if perform_fft else 'default')
    
    param_grid = {
        'C': [1, 3, 10, 30, 100]
    }
    
    estimator = SVC(kernel=kernel, max_iter=5000)
    evaluate = _create_evaluator(estimator, param_grid, 'accuracy', N=20)
    
    return evaluate(X, y)


def _evaluate_handover_mlp(item, signal_type, perform_fft):

    X, y = _load_handover(item, signal_type, 'fft' if perform_fft else 'default')
    
    param_grid = {
        'learning_rate_init': [0.01, 0.03, 0.1, 0.3],
        'alpha': [0.0001, 0.001]
    }
    
    estimator = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000)
    evaluate = _create_evaluator(estimator, param_grid, 'accuracy', N=20)
    
    return evaluate(X, y)


def _evaluate_handover_rnn(item, signal_type, device):

    X, y = _load_handover(item, signal_type, 'tensor')
    
    param_grid = { 'lr': [0.001, 0.003, 0.01] }
    
    estimator = NeuralNetRegressor(RNNModule,
                                   module__input_dim=X.shape[2],
                                   module__output_dim=1,
                                   iterator_train__shuffle=True,
                                   max_epochs=1000,
                                   train_split=False,
                                   device='cuda:0',
                                   verbose=1)
    
    evaluate = _create_evaluator(estimator,
                                 param_grid,
                                 'neg_mean_absolute_error',
                                 ShuffleSplit(n_splits=1, test_size=.25))
    
    return evaluate(X, y)


#    ______              _    ______                      _                      _       
#   |  ____|            | |  |  ____|                    (_)                    | |      
#   | |__ ___   ___   __| |  | |__  __  ___ __   ___ _ __ _ _ __ ___   ___ _ __ | |_ ___ 
#   |  __/ _ \ / _ \ / _` |  |  __| \ \/ / '_ \ / _ \ '__| | '_ ` _ \ / _ \ '_ \| __/ __|
#   | | | (_) | (_) | (_| |  | |____ >  <| |_) |  __/ |  | | | | | | |  __/ | | | |_\__ \
#   |_|  \___/ \___/ \__,_|  |______/_/\_\ .__/ \___|_|  |_|_| |_| |_|\___|_| |_|\__|___/
#                                        | |                                             


def _load_food(signal_type, transformation):
    
    classes = [ 'empty', 'water', 'tofu', 'watermelon', 'banana', 'apple', 'pepper' ]
    
    base_signal_type = signal_type if signal_type != 'neuhalf' else 'neutouch'
    
    npzfile0 = np.load(f'preprocessed/{base_signal_type}_food_empty.npz')
    npzfile1 = np.load(f'preprocessed/{base_signal_type}_food_water.npz')
    npzfile2 = np.load(f'preprocessed/{base_signal_type}_food_tofu.npz')
    npzfile3 = np.load(f'preprocessed/{base_signal_type}_food_watermelon.npz')
    npzfile4 = np.load(f'preprocessed/{base_signal_type}_food_banana.npz')
    npzfile5 = np.load(f'preprocessed/{base_signal_type}_food_apple.npz')
    npzfile6 = np.load(f'preprocessed/{base_signal_type}_food_pepper.npz')
    
    X = np.vstack((npzfile0['signals'],
                   npzfile1['signals'],
                   npzfile2['signals'],
                   npzfile3['signals'],
                   npzfile4['signals'],
                   npzfile5['signals'],
                   npzfile6['signals']))
    
    y = np.concatenate((npzfile0['labels'] * 0,
                        npzfile1['labels'] * 1,
                        npzfile2['labels'] * 2,
                        npzfile3['labels'] * 3,
                        npzfile4['labels'] * 4,
                        npzfile5['labels'] * 5,
                        npzfile6['labels'] * 6)).astype(np.compat.long)
    
    if signal_type == 'biotac':
        
        X = X / 1000
    
    if signal_type == 'neutouch':
        
        X = X / 40
    
    if signal_type == 'neuhalf':
        
        X = X[:, 0:40, :] / 40
        
    if transformation == 'default':
        
        X = np.reshape(X, (X.shape[0], -1))
    
    if transformation == 'fft':
        
        X = np.abs(np.fft.fft(X)) / 10
        X = np.reshape(X, (X.shape[0], -1))
    
    if transformation == 'tensor' and signal_type == 'biotac':
    
        X = torch.Tensor(np.swapaxes(X, 1, 2))
        y = torch.Tensor(np.reshape(y, (-1, 1)))
    
    if transformation == 'tensor' and signal_type == 'neutouch':
        
        torch.Tensor(np.expand_dims(X, 2))
        y = torch.Tensor(np.reshape(y, (-1, 1)))
    
    return X, y


def _evaluate_food_svm(signal_type, perform_fft, kernel):

    X, y = _load_food(signal_type, 'fft' if perform_fft else 'default')
    
    param_grid = {
        'C': [1, 3, 10, 30, 100]
    }
    
    estimator = SVC(kernel=kernel, max_iter=5000)
    evaluate = _create_evaluator(estimator, param_grid, 'accuracy', N=20)
    
    return evaluate(X, y)


def _evaluate_food_mlp(signal_type, perform_fft):

    X, y = _load_food(signal_type, 'fft' if perform_fft else 'default')
    
    param_grid = {
        'learning_rate_init': [0.01, 0.03, 0.1, 0.3],
        'alpha': [0.0001, 0.001]
    }
    
    estimator = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000)
    evaluate = _create_evaluator(estimator, param_grid, 'accuracy', N=20)
    
    return evaluate(X, y)


def _evaluate_food_rnn(signal_type, device):

    X, y = _load_food(signal_type, 'tensor')
    
    param_grid = { 'lr': [0.001, 0.003, 0.01] }
    
    estimator = NeuralNetRegressor(RNNModule,
                                   module__input_dim=X.shape[2],
                                   module__output_dim=1,
                                   iterator_train__shuffle=True,
                                   max_epochs=1000,
                                   train_split=False,
                                   device='cuda:0',
                                   verbose=1)
    
    evaluate = _create_evaluator(estimator,
                                 param_grid,
                                 'neg_mean_absolute_error',
                                 ShuffleSplit(n_splits=1, test_size=.25))
    
    return evaluate(X, y)


#     _____ _      _____    _____       _             __               
#    / ____| |    |_   _|  |_   _|     | |           / _|              
#   | |    | |      | |      | |  _ __ | |_ ___ _ __| |_ __ _  ___ ___ 
#   | |    | |      | |      | | | '_ \| __/ _ \ '__|  _/ _` |/ __/ _ \
#   | |____| |____ _| |_    _| |_| | | | ||  __/ |  | || (_| | (_|  __/
#    \_____|______|_____|  |_____|_| |_|\__\___|_|  |_| \__,_|\___\___|
#                                                                      


if __name__ == '__main__':
    
    import sys
    
    if len(sys.argv) == 2: evaluate(sys.argv[1])
    if len(sys.argv) == 3: evaluate(sys.argv[1], sys.argv[2])