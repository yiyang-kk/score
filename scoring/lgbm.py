import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import gc
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
from hyperopt import hp, Trials, tpe, fmin, STATUS_OK, space_eval
import shap

import warnings
warnings.filterwarnings('ignore')

def comp_gini(target,predictions,sample_weight=None):
    """computes gini
    
    Args:
        target (np.array): true target
        predictions (np.array): target predictions
        weight (np.array, optional): observation weight (default: None)
    """

    return(2 * roc_auc_score(target, predictions, sample_weight=sample_weight) - 1)

    


def show_plot_roc(fpr,tpr, gini, label):
    """Plots ROC curve(s) for both: CV model or single model
    
    Arguments:
        fpr (list of float): list of false positive rates
        tpr (list of float): list of true positive rates
        gini (list of float): list of ginis
        label (str): label of the set ('train' or 'valid') are options used in code
    """
    
    
    plt.figure(figsize=(10,5))

    for i in range(len(gini)):

            plt.plot(fpr[i], tpr[i], label = 'ROC curve of {} {}. fold (GINI = {})'.format(label, i, round(gini[i],3)))
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()


class LGBM_model:
    """LightGBM wrapper with methods suitable for credit scoring.
    
    Args:
        cols_pred (list of str): list of predictors
        params (dictionary): parameters of lgb.train() function
        use_CV (bool, optional): True - train n-fold CV with merged train and valid sets (default: {False})
        CV_folds (int, optional): In case of True, number of CV folds (default: {3})
        CV_seed (int, optional): In case of True, seed for k-fold split (default: {98765})
    """

    def __init__(self, cols_pred, params, use_CV=False, CV_folds=3, CV_seed=98765):

        self.cols_pred = cols_pred
        self.use_CV = use_CV

        if self.use_CV == False:
            self.CV_folds = 1
        else:
            self.CV_folds = CV_folds

        self.CV_seed = CV_seed
        self.params = params

        self.cols_cat = None
        self.explainer = None
        self.set_to_shap = None
        self.train = None
        self.shap_values = None
        self.base_value = None



    def __one_model(self, x_train, x_valid, y_train, y_valid, w_train=None, w_valid=None):
        """ Training of single lgbm model

        Args:
            x_train (pandas.DataFrame): training set
            x_valid (pandas.DataFrame): valid set
            y_train (pandas.DataFrame): target of training set
            y_valid (pandas.DataFrame): target of valid set
            w_train (pandas.DataFrame, optional): training set weight (default: None)
            w_valid (pandas.DataFrame, optional): valid set weight (default: None)
        
        Returns:
            lgbm.Booster: lgbm booster model
        """

        dtrain = lgb.Dataset(x_train, label=y_train, weight=w_train)
        dvalid = lgb.Dataset(x_valid, label=y_valid, weight=w_valid)

        evals = {}

        booster = lgb.train(
            params = self.params,
            train_set = dtrain,
            valid_sets = [dvalid, dtrain],
            verbose_eval = 200,
            evals_result = evals
        )

        self.evals = evals
        self.final_iterations = booster.current_iteration()

        return booster


    def show_progress(self):
        """Show curve with AUC value progress during model training iterations.
        """

        try:
            evals = self.evals
            final_iterations = self.final_iterations
        except:
            print('Model has not been trained yet.')
            return
            
        print('Progress during most recent training of model.')
        plt.figure(figsize=(8,8))
        plt.title('loss curve')
        for k, v in evals.items():
            plt.plot(np.arange(1, len(v['auc'])+1),  v['auc'], label=k)
        plt.axvline(final_iterations, color='grey', linestyle='--')
        plt.xlabel('Iteration')
        plt.ylabel('auc')
        plt.legend(loc="lower right")
        plt.show()

        return


    def fit_model(self, x_train, x_valid, y_train, y_valid, w_train=None, w_valid=None):
        """ Fitting of a model
            IF use_CV=False
                Fits model from train set and training is stopped when it starts to overfit on valid set

            IF use_CV=True
                Merges train and valid sets and fits n-time CV


        Args:
            x_train (pandas.DataFrame): training set
            x_valid (pandas.DataFrame): valid set
            y_train (pandas.DataFrame): target of training set
            y_valid (pandas.DataFrame): target of valid set
            w_train (pandas.DataFrame, optional): training set weight (default: None)
            w_valid (pandas.DataFrame, optional): valid set weight (default: None)
        
        Returns:
            list of lgbm.Booster: 
                IF use_CV = False
                    list with one lgb.booster model
                IF use_CV = True
                    list with n lgb.booster models 
        """


        x_train = x_train[self.cols_pred]
        x_valid  = x_valid[self.cols_pred]



        if self.use_CV == False:

            print('I am not using CV option, training is stopped when model starts to overfit on valid set')
            print(' ')

            models = self.__one_model( x_train, x_valid, y_train, y_valid, w_train, w_valid)

            fpr_train, tpr_train, _ = roc_curve(y_train, models.predict(x_train), sample_weight=w_train )
            gini_train = comp_gini(y_train, models.predict(x_train), sample_weight=w_train )

            fpr_valid, tpr_valid, _ = roc_curve(y_valid, models.predict(x_valid), sample_weight=w_valid )
            gini_valid = comp_gini(y_valid, models.predict(x_valid), sample_weight=w_valid )

            fpr_train = [fpr_train]
            tpr_train = [tpr_train]
            gini_train = [gini_train]

            fpr_valid = [fpr_valid]
            tpr_valid = [tpr_valid]
            gini_valid = [gini_valid]

            models=[models]


        else:
            # merging train and valid dataset

            print('I am using CV option, train and valid sets are merged')
            print(' ')


            merged_ds = pd.concat([x_train, x_valid]).reset_index(drop=True)
            merged_target = pd.concat([y_train, y_valid]).reset_index(drop=True)
            if (w_train is not None) and (w_valid is not None):
                merged_weight = pd.concat([w_train, w_valid]).reset_index(drop=True)
            else:
                merged_weight = None

            folds = KFold(n_splits=self.CV_folds)
            models = []

            fpr_train  = []
            fpr_valid   = []
            tpr_train  = []
            tpr_valid   = []
            gini_train = []
            gini_valid  = [] 

            for train_index, valid_index in folds.split(merged_ds):
                X_train, X_valid =  merged_ds.loc[train_index, :], merged_ds.loc[valid_index, :] 
                Y_train, Y_valid =  merged_target.loc[train_index], merged_target.loc[valid_index]
                if merged_weight:
                    W_train, W_valid =  merged_weight.loc[train_index], merged_weight.loc[valid_index]
                else:
                    W_train, W_valid =  None, None
                
                model= self.__one_model( X_train, X_valid, Y_train, Y_valid, w_train=W_train, w_valid=W_valid)
                models.append(model)

                fpr, tpr, _ = roc_curve(Y_train, model.predict(X_train), sample_weight=W_train)
                fpr_train.append(fpr)
                tpr_train.append(tpr)
                gini_train.append(comp_gini(Y_train, model.predict(X_train), sample_weight=W_train))

                fpr, tpr, _ = roc_curve(Y_valid, model.predict(X_valid), sample_weight=W_valid)
                fpr_valid.append(fpr)
                tpr_valid.append(tpr)
                gini_valid.append(comp_gini(Y_valid, model.predict(X_valid), sample_weight=W_valid))
  
        show_plot_roc(fpr_train, tpr_train, gini_train, 'train')

        show_plot_roc(fpr_valid, tpr_valid, gini_valid, 'valid')

        self.models = models

        return self.models

    def __comp_var_imp(self, models=None):  

        """Creates dataframe with variable importances. In case of use_CV = True than compute average var. imp. based on all models

        Args:
            models (list of lgbm.Booster, optional): (default: None)
        
        Returns:
            pandas.DataFrame: df with gain and weight importance for all features
        """

        if (models is None) and (not hasattr(self, 'models')):
            raise RuntimeError('Model was not trained yet.')
        elif (models is None):
            models = self.models

        importance_df = pd.DataFrame()
        importance_df["Feature"] = self.cols_pred 
        importance_df["importance_gain"] = 0
        importance_df["importance_weight"] = 0

        for model in models:
            importance_df["importance_gain"] = importance_df["importance_gain"] + model.feature_importance(importance_type = 'gain') / self.CV_folds
            importance_df["importance_weight"] = importance_df["importance_weight"] + model.feature_importance(importance_type = 'split') / self.CV_folds

        return importance_df

    def plot_imp(self , models=None, imp_type='importance_gain', ret=False, show=True, n_predictors=100): 
        """takes output of __comp_var_imp and print top n predictors in nice form, sorted by highest importance
                
        Args:
            models (list of lgbm.Booster, optional): list of lgbm.boosters
            imp_type (string, optional): 'importance_gain' or 'importance_weight'
            ret (bool, optional): True for return pandas.Dataframe with features a importances   (default: {False})
            show (bool, optional): True for ploting feature importance (default: {True})
            n_predictors (int, optional): number of best features (default: {100})
        
        Raises:
            ValueError: if 'imp_type' parameter is different from 'importance_gain' or 'importance_weight'
        
        Returns:
            pandas.Dataframe: df  with features a importances            
        """

        if (models is None) and (not hasattr(self, 'models')):
            raise RuntimeError('Model was not trained yet.')
        elif (models is None):
            models = self.models

        if ((imp_type != 'importance_gain') & (imp_type != 'importance_weight')):
            raise ValueError('Only "importance_gain" and "importance_weight" are possible imp_types.')

        dataframe = self.__comp_var_imp( models)

        if show == True:
            plt.figure(figsize = (20,n_predictors/2))
            sns.barplot(x=imp_type, y="Feature", data=dataframe.sort_values(by=imp_type, ascending=False).head(n_predictors))
            plt.show()

        if ret == True:
            return dataframe.sort_values(by=imp_type, ascending=False).head(n_predictors)[['Feature', imp_type ]]

    def predict(self , set_to_predict, models=None):
        """Predicts probabilities on new dataset
        
        Args:
            set_to_predict (pandas.DataFrame): set for which you want to predict ouput
            models (list of lgbm.Booster, optional): list of lgbm.boosters
        
        Returns:
            numpy.array: predicted probabilities
        """

        if (models is None) and (not hasattr(self, 'models')):
            raise RuntimeError('Model was not trained yet.')
        elif (models is None):
            models = self.models

        predictions = np.zeros(set_to_predict.shape[0])

        for model in models:
            predictions = predictions + model.predict(set_to_predict[self.cols_pred]) / self.CV_folds

        return predictions

    def marginal_contribution(self, x_train, x_valid, y_train, y_valid, set_to_test, set_to_test_target, w_train=None, w_valid=None, set_to_test_weight=None, silent=False):
        """Computes gini performance of model on set_to_test, trained without particular feature. 
            This is computed for every feature separately from self.cols_pred
        
        Args:
            x_train (pandas.DataFrame): training set
            x_valid (pandas.DataFrame):valid set
            y_train (pandas.DataFrame): target of training set
            y_valid (pandas.DataFrame): target of valid set
            set_to_test (pandas.DataFrame): set on which you want to compute marginal contribution
            set_to_test_target (pandas.DataFrame): target of set on which you want to compute marginal contribution
            w_train (pandas.DataFrame, optional):training set weights (default: None)
            w_valid (pandas.DataFrame, optional): valid set weights (default: None)
            set_to_test_weight (pandas.DataFrame, optional): set on which you want to compute marginal contribution weights (default: None)
            silent (boolean, optional): whether the output should NOT be displayed on screen (default: False)
        
        Returns:
            pandas.DataFrame: dataframe with 4 columns - feature, gini with feature, gini without feature and difference of gini with feature and gini without feature
        """

        diff_dataframe = pd.DataFrame()
        diff_dataframe = pd.DataFrame(columns=['Feature','Perf_with','Perf_without','Difference'])
        diff_dataframe = diff_dataframe.fillna(0)

        predictors=self.cols_pred

        model = self.__one_model(x_train[predictors], x_valid[predictors], y_train, y_valid, w_train=w_train, w_valid=w_valid) 
        
        pred_set_to_test = model.predict(set_to_test[predictors])      
        gini_test_original = comp_gini(set_to_test_target, pred_set_to_test, sample_weight=set_to_test_weight)

        j = 0       
        for i, pred in enumerate(predictors):
            predictors_new = predictors.copy()
            predictors_new.remove(pred)

            if 'monotone_constraints' in self.params:
                monotone_con_original = self.params['monotone_constraints']
                self.params['monotone_constraints'] = self.params['monotone_constraints'][:i] + self.params['monotone_constraints'][i+1:]

            model = self.__one_model(x_train[predictors_new], x_valid[predictors_new], y_train, y_valid, w_train=w_train, w_valid=w_valid) 
            pred_set_to_test = model.predict(set_to_test[predictors_new])

            if 'monotone_constraints' in self.params:
                self.params['monotone_constraints'] = monotone_con_original

            diff_dataframe.loc[j,'Feature'] = pred
            diff_dataframe.loc[j,'Perf_with'] = gini_test_original*100
            diff_dataframe.loc[j,'Perf_without'] = comp_gini(set_to_test_target, pred_set_to_test, sample_weight=set_to_test_weight)*100
            diff_dataframe.loc[j,'Difference'] = (diff_dataframe.loc[j,'Perf_with']-diff_dataframe.loc[j,'Perf_without']) 
        
            j+=1

        if not silent:
            print(diff_dataframe.sort_values(by=['Difference']).to_string())

        return diff_dataframe.sort_values(by=['Difference'])


    def print_shap_values(self, cols_num, cols_cat, x_train, x_valid, y_train, y_valid, set_to_shap, w_train=None, w_valid=None, output_folder=None):
        """This method computes shap values for given set_to_shap
        
        Args:
            cols_num (list of str): names of numerical columns
            cols_cat (list of str): names of categorical columns
            x_train (pandas.DataFrame): training set
            x_valid (pandas.DataFrame): valid set
            y_train (pandas.DataFrame): target of training set
            y_valid (pandas.DataFrame): target of valid set
            set_to_shap (pandas.DataFrame): set on which you want to compute shap values
            output_folder (str): folder to output charts
            w_train (pandas.DataFrame, optional): training set weights (default: None)
            w_valid (pandas.DataFrame, optional): valid set weights (default: None)
            output_folder (str, optional): where the pictures should be saved ... if None, they will be displayed instead (default: None)

        Returns:
            pandas.DataFrame: Dataframe with name of the feature and its shap value 
        """

        print('Model has to be trained again because of categorical variables encoding')
        print(' ')
        
        train = x_train.copy()
        x_train = x_train[self.cols_pred]
        x_valid = x_valid[self.cols_pred]
        set_to_shap = set_to_shap[self.cols_pred]
        self.cols_cat = cols_cat


        for col in cols_cat:
            x_train[col] = x_train[col].cat.add_categories('NA').fillna('NA')
            x_valid[col] = x_valid[col].cat.add_categories('NA').fillna('NA')
            set_to_shap[col] = set_to_shap[col].cat.add_categories('NA').fillna('NA')
            _ , indexer = pd.factorize(x_train[col])
            x_train[col] = indexer.get_indexer(x_train[col])
            x_valid[col] = indexer.get_indexer(x_valid[col])
            set_to_shap[col] = indexer.get_indexer(set_to_shap[col])


        model=self.__one_model(x_train, x_valid, y_train, y_valid, w_train=w_train, w_valid=w_valid)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(set_to_shap)

        self.explainer = explainer
        self.set_to_shap = set_to_shap
        self.train = train

        if isinstance(shap_values, list):
        
        #this if - else covers problem discussed here
        #https://github.com/slundberg/shap/issues/526
        #explainer.shap_values() behaves differently on a local laptop and on server        
        
            self.shap_values = shap_values[1]
            self.base_value = explainer.expected_value[1]

        else:
            self.shap_values = shap_values
            self.base_value = explainer.expected_value

        if output_folder is None:
            shap.summary_plot(self.shap_values, set_to_shap)
            shap.summary_plot(self.shap_values, set_to_shap, plot_type='bar')
        else:
            shap.summary_plot(self.shap_values, set_to_shap, show=False)
            plt.savefig(output_folder+'/shap.png', bbox_inches='tight')
            plt.close()
            shap.summary_plot(self.shap_values, set_to_shap, plot_type='bar', show=False)
            plt.savefig(output_folder+'/shap_abs.png', bbox_inches='tight')
            plt.close()
 
        var_imp_dataframe = {'Feature': self.cols_pred, 'Shap_importance': np.mean(abs(self.shap_values),axis=0) }

        return pd.DataFrame(var_imp_dataframe).sort_values(by=['Shap_importance'], ascending = False)
        

        

    def print_shap_interaction_matrix(self):
        """Prints shap interaction matrix, based on https://christophm.github.io/interpretable-ml-book/shap.html#shap-interaction-values
            It prints sum of absolute interactions values throught all observations.
            Diagonal values are manually set to zero.

        """

        shap_inter_values = self.explainer.shap_interaction_values(self.set_to_shap)

        corr = np.sum(abs(shap_inter_values),axis=0)
        corr[np.diag_indices_from(corr)] = 0

        corr = pd.DataFrame(corr, columns=self.cols_pred, index=self.cols_pred)
        corr = round(corr,0)

        sns.set(rc={'figure.figsize':(20,20)})
        sns.heatmap(corr, annot=True, xticklabels=self.cols_pred, yticklabels=self.cols_pred, fmt='.5g')


    def shap_one_row(self, row):
        """Prints decision plot for 1 row.
        
        Args:
            row (int): row number of self.set_to_shap
        """

        shap.force_plot(self.base_value, self.shap_values[row,:],self.set_to_shap.iloc[row,:], link='identity', matplotlib=True)

        shap.force_plot(self.base_value, self.shap_values[row,:],self.set_to_shap.iloc[row,:], link='logit', matplotlib=True)


    def shap_dependence_plot(self, x, y=None):
        """Prints shap dependence plot for given feature.
            If y is not specified, algorithm finds it automatically.
        
        Args:
            x (str): feature name
            y (str): other feature name (default: {None})
        """

        if x in self.cols_cat:         
            print('Encoding of categories for your variable')   
            labels, uniques = pd.factorize(self.train[x].cat.add_categories('NA').fillna('NA'))
            for j in range(len(np.unique(labels))):
                print(np.unique(labels)[j])
                print(uniques[j])


        if  y is None:
            shap.dependence_plot(x, self.shap_values, self.set_to_shap)

        else:            
            shap.dependence_plot(x, self.shap_values, self.set_to_shap, interaction_index = y)


    
    def param_hyperopt(self, x_train, x_valid, y_train, y_valid, w_train=None, w_valid=None, n_iter = 500, space = None):
        """Finds optimal hyperparameters based on cross validation AUC
        
        Args:
            x_train (pandas.DataFrame): training set
            x_valid (pandas.DataFrame): valid set
            y_train (pandas.DataFrame): target of training set
            y_valid (pandas.DataFrame): target of valid set
            w_train (pandas.DataFrame, optional): training set weights (default: None)
            w_valid (pandas.DataFrame, optional): valid set weights (default: None)
            n_iter (int, optional): number of iteration (default: {500})
            space (dict, optional): hyperparameter space to be searched. if None, a default space is used (deafult: None)
        
        Returns:
            dict: optimal hyperparameters
        """

        merged_ds = pd.concat([x_train[self.cols_pred], x_valid[self.cols_pred]]).reset_index(drop=True)
        merged_target = pd.concat([y_train, y_valid]).reset_index(drop=True)
        if (w_train is not None) and (w_valid is not None):
            merged_weight = pd.concat([w_train, w_valid]).reset_index(drop=True)
        else:
            merged_weight = None

        train_data = lgb.Dataset(merged_ds, label=merged_target, weight=merged_weight)

        def objective (params):

            print(params)
            cv_results = lgb.cv(params, train_data, stratified = True, nfold = 3)

            best_score = -max(cv_results['auc-mean'])
            print('Actual gini:')
            print(2*abs(best_score)-1)
            print('----------------------------------------------------------------------------------')

            return {'loss': best_score, 'params': params, 'status': STATUS_OK}

        if space is None:
            space = {
                'learning_rate': hp.choice('learning_rate', np.arange(0.02, 0.1, 0.02, dtype=float)),
                'num_leaves': hp.choice('num_leaves', np.arange(2, 64, 2, dtype=int)),
                'max_depth': hp.choice('max_depth', np.arange(2, 5, 1, dtype=int)),
                'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.5, 0.9, 0.05, dtype=float)),
                'subsample': hp.choice('subsample', np.arange(0.5, 0.9, 0.05, dtype=float)),
                'min_data_in_leaf': hp.choice('min_data_in_leaf', np.arange(50, 500, 50, dtype=int)),
                'min_child_weight': hp.choice('min_child_weight', np.arange(10, 100, 10, dtype=int)),
                'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
                'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
                'verbose':1,
                'metric':'auc',
                'objective':'binary',
                'early_stopping_rounds':50,
                'num_boost_round':100000,
                'seed':1234
                }


        tpe_algorithm = tpe.suggest
        bayes_trials = Trials()

        best = fmin(fn = objective, space = space, algo = tpe_algorithm, 
            max_evals = n_iter, trials = bayes_trials)

        best_values = space_eval(space, best)

        print('Best combination of parameters is:')
        print(best_values)
        print('')
        return best_values
