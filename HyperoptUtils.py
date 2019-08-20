"""
Programmer: Francis Motta, edited by Chris Tralie
"""
import csv
import time
import pickle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd

# hyperparameter optimization routines
from hyperopt import hp
from hyperopt import tpe
from hyperopt import fmin
from hyperopt import Trials
from hyperopt import STATUS_OK
from hyperopt.pyll.stochastic import sample

# persistence images routines
import PersistenceImages.persistence_images as pimgs

from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os.path

ITER = 0

def vec_dgm_by_per(dgm, start=0, num_pairs=50, per_only=True):
    """
    Generates a simple feature vector of length num_pairs or 2*(num_pairs) from a persistence diagram that consists
    of either the persistence or the birth & death values of the `start` through the `start+num_pairs` most persistent pairs.
    :param dgm: (N,2) numpy array encoding a persistence diagram
    :param start: non-negative integer specifying the first most persistence pair to consider 
    :param num_pairs: positive integer specifying the number of persistence pairs to consider
    :param per_only: If True, the end-start coordinates encode only the persistence of each pair, otherwise the 2*(end-start) 
               coordinates encode the birth and death of each pair.
    """  
    dgm = np.copy(dgm)
    N = dgm.shape[0]
    
    pers = dgm[:, 1]-dgm[:, 0]
    
    # make sure we have valid indices
    start=int(start)
    end=int(start+num_pairs)
    start = min(start, N)
    end = min(end, N)

    ind = pers.argsort()[::-1]
    dgm = dgm[ind, :]
    pers = pers[ind]

    if per_only:
        ret = np.zeros(num_pairs)
        ret[0:end-start] = pers[start:end]
        return ret
    else:
        ret = np.zeros((num_pairs, 2))
        ret[0:end-start] = dgm[start:end, :]
        return ret.flatten(order='C')

def vec_dgm_by_per_images(dgm, birth_range=None, pers_range=None, pixel_size=None, weight=pimgs.weighting_fxns.persistence, weight_params=None, kernel=pimgs.kernels.bvncdf, kernel_params=None, skew=True):
    """
    Generates a flattened persistence image feature vector from a persistence diagram subject to the specified hyperparameters
    :param dgm: (N,2) numpy array encoding a persistence diagram
    :param birth_range: tuple specifying lower and upper birth value of the persistence image
    :param pers_range: tuple specifying lower and upper persistence value of the persistence image
    :param pixel_size: size of square pixel
    :param weight: function to weight the birth-persistence plane
    :param weight_params: arguments needed to specify the weight function
    :param kernel: cumulative distribution function of kernel
    :param kernel_params: arguments needed to specify the kernel (cumulative distribution) function
    :param skew: boolean flag indicating if diagram needs to be converted to birth-persistence coordinates
                     (default: True)
    :return: flattened persistence image in row-major (C-style) order
    """
    dgm = np.copy(dgm)
    
    # setup the imager object with specified parameters
    imgr = pimgs.PersistenceImager(birth_range=birth_range, 
                                   pers_range=pers_range, 
                                   pixel_size=pixel_size,
                                   weight=weight, 
                                   weight_params=weight_params, 
                                   kernel=kernel, 
                                   kernel_params=kernel_params)
    
    # generate and return the persistence image
    img = imgr.transform(dgm, skew=skew)
    return img.flatten(order='C')

def gen_dgm_feature_df(dgm_df, method):
    """
    For each design generate a persistence diagram feature vector and store in a feature-array dataframe.
    dgm_df - dataframe of designs that has the column 'dgm' containing the dictionaries of
             persistence diagrams for each design
    method - a python function which takes as input a (n,2)-ndarray encoding a dgm and returns a (k,)-ndarray 
             feature vector
    """
    # extract dimensions from the diagram dataframe
    dims = dgm_df.iloc[0, :].dgm.keys()
        
    num_designs = len(dgm_df)
    dgms = dgm_df.dgm
    
    # populate the feature-array dataframe
    for i in range(num_designs):
        # vectorize diagrams in each dimension and concatenate
        temp_vec = []
        for dim in dims:
            temp_vec = np.concatenate((temp_vec, method(dgms.iloc[i][dim])))
        
        if i == 0:
            feature_array = np.empty((len(dgm_df), len(temp_vec)), dtype=np.float64)
            feature_array[i, :] = temp_vec
        else:
            feature_array[i, :] = temp_vec
    
    dgm_vec_df = pd.DataFrame(data=feature_array, index=dgm_df.index)
    dgm_vec_df.columns = ['feature' + str(col) for col in dgm_vec_df.columns.values]
    
    return dgm_vec_df

def lotocv_objective(params, dgm_df, target_df, scorer, verbose=False):
    """
    A model-validation function which can be minimized using Bayesian optimization. The input is a dictionary specifying 
    a choice of parameters, the output is the mean score of the leave-one-topology-out cross-validation model.
    
    params - A nested dictionary of parameter values containing both of the keys
               'estimator_params' - a dictionary of parameters defining the estimator and its hyperparameters
               'dgm_vec_params' - a dictionary of parameters defining the diagram vectorization method and hyperparameters
               
               Each dictionary value must contain the key 'method' specifying the particular models to use:
                 params['estimator_params'][method] - an estimator handle which implements the .fit() 
                                                      and .predict() methods (e.g. sklearn.svm.SVC)
                 params['dgm_vec_params'][method] - a python function handle which takes as input a (n,2)-ndarray encoding 
                                                    a diagram and returns a (k,)-ndarray feature vector
                
             Any additional parameters needed to specify the estimator or the vectorization method should be passed via
             a dictionary of argument-value pairs stored in the value of the key 'kwargs'
             (e.g. params['estimator_params']['kwargs'] = {'C': 1.0, 'kernel': 'poly', 'gamma': 0.4})
             
    dgm_df - dataframe with column 'dgm' that contains the dictionaries of persistence diagrams for each design

    target_df - dataframe with target value(s) to be predicted by the estimator
    
    scorer - a function which takes real target values and predicted target values and returns a score measuring the accuracy 
             of the prediction (e.g. sklearn.metrics.roc_auc_score)
    
    verbose - flag indicating if progress should be displayed
    
    ** NOTE: The indices of `dgm_df` and `target_df` are assumed to be sorted in the same order **
    ** NOTE: `dgm_df`, `target_df`, `scorer`, and `fixed_params` are not part of the optimization search space and will be 
             fixed during the optimization process while the parameters in `params` are allowed to vary **
    """
    starttime = time.time()
    # ---------------------
    
    # parse all parameters and ensure they yield valid mappings
    estimator_method = params['estimator_params']['method']  # required
    dgm_vec_method = params['dgm_vec_params']['method']  # required
    
    estimator_kwargs = params['estimator_params'].get('kwargs')  # optional
    if estimator_kwargs is None:
        estimator_kwargs = {}
       
    dgm_vec_kwargs = params['dgm_vec_params'].get('kwargs')  # optional
    if dgm_vec_kwargs is None:
        dgm_vec_kwargs = {}
        
    # ---------------------
    
    # 1.) feature generation step
    # pass the optional vectorization parameters to the vectorization method
    _dgm_vec_method = lambda dgm: dgm_vec_method(dgm, **dgm_vec_kwargs)
    
    # generate the feature vector dataframe
    dgm_feature_df = gen_dgm_feature_df(dgm_df, _dgm_vec_method)
    dgm_feature_df['group_index'] = dgm_df['group_index']
    
    # ---------------------
    
    # 2.) model validation step
    # pass the optional estimator parameters to the estimator method
    estimator_method = estimator_method.set_params(**estimator_kwargs)
    
    # perform leave-one-topology-out cross-validation with the estimator and compute loss
    lotocv_dict = lotocv(dgm_feature_df, target_df, estimator_method, scorer=scorer, verbose=verbose)
    loss = 1 - sum(lotocv_dict.values()) / len(lotocv_dict)

    # -----------------------
    endtime = time.time()

    return {'loss': loss, 'status': STATUS_OK, 'params': params, 'eval_time': endtime-starttime, 'iteration': ITER}


def cv_objective(params, dgm_df, target_df, scorer, cv=10, verbose=False):
    """
    A model-validation function which can be minimized using Bayesian optimization. The input is a dictionary specifying 
    a choice of parameters, the output is the mean score of cv-fold cross-validation across all samples
    
    params - A nested dictionary of parameter values containing both of the keys
               'estimator_params' - a dictionary of parameters defining the estimator and its hyperparameters
               'dgm_vec_params' - a dictionary of parameters defining the diagram vectorization method and hyperparameters
               
               Each dictionary value must contain the key 'method' specifying the particular models to use:
                 params['estimator_params'][method] - an estimator handle which implements the .fit() 
                                                      and .predict() methods (e.g. sklearn.svm.SVC)
                 params['dgm_vec_params'][method] - a python function handle which takes as input a (n,2)-ndarray encoding 
                                                    a diagram and returns a (k,)-ndarray feature vector
                
             Any additional parameters needed to specify the estimator or the vectorization method should be passed via
             a dictionary of argument-value pairs stored in the value of the key 'kwargs'
             (e.g. params['estimator_params']['kwargs'] = {'C': 1.0, 'kernel': 'poly', 'gamma': 0.4})
             
    dgm_df - dataframe with column 'dgm' that contains the dictionaries of persistence diagrams for each sample

    target_df - dataframe with target value(s) to be predicted by the estimator
    
    scorer - a function which takes real target values and predicted target values and returns a score measuring the accuracy 
             of the prediction (e.g. sklearn.metrics.roc_auc_score)
    
    cv - number of train/test cross-validation splits to perform
    
    verbose - flag indicating if progress should be displayed
    
    ** NOTE: The indices of `dgm_df` and `target_df` are assumed to be sorted in the same order **
    ** NOTE: `dgm_df`, `target_df`, `scorer`, and `fixed_params` are not part of the optimization search space and will be 
             fixed during the optimization process while the parameters in `params` are allowed to vary **
    """   
    # track global iteration
    global ITER
    ITER=ITER+1
    starttime = time.time()
    # ---------------------
    
    # parse all parameters and ensure they yield valid mappings
    estimator_method = params['estimator_params']['method']  # required
    dgm_vec_method = params['dgm_vec_params']['method']  # required
    
    estimator_kwargs = params['estimator_params'].get('kwargs')  # optional
    if estimator_kwargs is None:
        estimator_kwargs = {}
       
    dgm_vec_kwargs = params['dgm_vec_params'].get('kwargs')  # optional
    if dgm_vec_kwargs is None:
        dgm_vec_kwargs = {}
        
    # ---------------------
    
    # 1.) feature generation step
    if verbose:
        print('Vectorizing Diagrams\n')
    # pass the optional vectorization parameters to the vectorization method
    _dgm_vec_method = lambda dgm: dgm_vec_method(dgm, **dgm_vec_kwargs)
    
    # generate the feature vector dataframe
    dgm_feature_df = gen_dgm_feature_df(dgm_df, _dgm_vec_method)
    
    # ---------------------
    
    # 2.) model validation step
    if verbose:
        print('Training and Testing Estimator\n')
    # pass the optional estimator parameters to the estimator method
    estimator_method = estimator_method.set_params(**estimator_kwargs)
    
    # perform cross-validation with the estimator and compute loss
    if verbose:
        cv_verbose = 2
    else:
        cv_verbose = 0
        
    scores = cross_validate(estimator_method, dgm_feature_df, target_df, scoring=scorer, cv=cv,
                            n_jobs=-1, 
                            verbose=cv_verbose)
    
    loss = sum(scores['test_score']) / len(scores['test_score'])

    # -----------------------
    endtime = time.time()

    return {'loss': loss, 'status': STATUS_OK, 'params': params, 'eval_time': endtime-starttime, 'iteration': ITER}

def get_bone_data_df():
    """
    Return the bone data as data frames
    """
    N = 18
    bone_data = [sio.loadmat("PDs/%i.mat"%i) for i in range(N)]
    vals = ['trabnum', 'trabtick', 'trablen', 'bv_tv']
    vals = {v:[bone_data[i][v].flatten()[0] for i in range(N)] for v in vals}
    vals['dgm'] = [{h:bone_data[i][h] for h in ['H1', 'H2']} for i in range(N)]
    return pd.DataFrame(vals)

if __name__ == '__main__':
    # -------------------
    # Perform a more complex Bayesian optimization search across persistence images,
    # as well as number of principal components to retain and various RF model hyperparameters to plot over-time improvement of 
    # loss
    bone_df = get_bone_data_df()
    dgm_df = bone_df[['dgm']]
    target_df = bone_df['trabnum']
    scorer = make_scorer(mean_squared_error)
    max_evals = 495
    cv=5

    # precompute the persistence image region over the full dataset
    birth_range = (0, 1)
    pers_range = (0, 1)
    
    pipeline_ridge = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge())])

    param_space = {'estimator_params': 
                    {
                    'method': pipeline_ridge,
                    'kwargs':  {
                                'ridge__normalize':False,
                                'ridge__fit_intercept':True,
                                'ridge__alpha':hp.loguniform('alphas', -10, 0)
                                }
                    },
                    'dgm_vec_params':
                    hp.choice('dgm_vec_params',
                    [
                    {
                    'method': vec_dgm_by_per_images,
                    'kwargs': {
                                'birth_range': birth_range,
                                'pers_range': pers_range,
                                'pixel_size': hp.uniform('pixel_size', 0.025, 1.0),
                                'weight_params': {'n': hp.uniform('n', 1, 3)},
                                'kernel_params': {'sigma': hp.uniform('sigma', 0, 0.2)}
                                }
                    }
                    ])
                }


    # ---------------------------------
    # create the objective function to minimize, passing in all fixed arguments
    objective = lambda params: cv_objective(params,
                                            dgm_df=dgm_df,
                                            target_df=target_df, 
                                            scorer=scorer,
                                            cv=cv,
                                            verbose=True)


    # continue parameterization run if already started
    if os.path.isfile('data/complex_vec_dgm_bayes_trials.pickle'):
        with open('data/complex_vec_dgm_bayes_trials.pickle','rb') as f:
            bayes_trials = pickle.load(f)
    else:
        bayes_trials = Trials()

    # run the hyperparamter optimization
    best = fmin(fn=objective, 
                space=param_space, 
                algo=tpe.suggest, 
                max_evals=max_evals, 
                trials=bayes_trials)
