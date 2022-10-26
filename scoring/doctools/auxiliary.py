from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

class RussianImputer(BaseEstimator, TransformerMixin):
   
    def __init__(self):
        self.miss_dict = {}
                 
    def fit(self, X, y):
        import numpy as np
        
        df = X.copy()
        if type(df) != 'pandas.core.frame.DataFrame':
            df = pd.DataFrame(df)   
        df['target'] = y
        
        for predictor in df.columns[:-1]:
            res = df[['target',predictor]].groupby(df[predictor].isnull()).apply(np.mean)
            if len(res) > 1: # has some missing
                dr_hit = res.iloc[0][0]
                dr_no_hit = res.iloc[1][0]
                mean_hit = res.iloc[0][1]
                print((dr_hit, dr_no_hit, mean_hit))
                assoc= df[['target',predictor]].corr().values[0,1]
                print(assoc)
                if assoc >= 0  :
                    print('positive')
                    self.miss_dict[predictor] = mean_hit * dr_no_hit / dr_hit
                else:
                    print('negative')
                    self.miss_dict[predictor] = 1-((1-mean_hit) * dr_no_hit / dr_hit)
            else:
                self.miss_dict[predictor] = df[predictor].mean()
        return self

    def transform(self, X):
        df = X.copy()
        if type(df) != 'pandas.core.frame.DataFrame':
            df = pd.DataFrame(df)  
        for predictor in df.columns:
            df[predictor] =  df[predictor].fillna(self.miss_dict[predictor])
        return df


class LinearRegressionImputer(BaseEstimator, TransformerMixin):
   
    def __init__(self):

        self.imput_value_dict = {}
        self.predict_missing = []
        self.table = pd.DataFrame()
        self.mean_score = []

    def fit(self,X,y,orig_bin_num=20):
            
        def logit(x):
            logit = np.log(x / (1-x))
            return logit

        def sgn(x):
            if x > 0:
                return 1
            elif x < 0:
                return -1
            else:
                return 0

        def delogit(x):
            delogit = 1 / (1 + np.exp(-x))
            return delogit
        
        df= X.copy()
        if type(df) != 'pandas.core.frame.DataFrame':
            df = pd.DataFrame(df) 
        df['targets']= y

        for cont_column  in df.columns[:-1]:
            hit_mask = ~df[cont_column].isnull()
            non_hit_mask = df[cont_column].isnull()

            score = df[hit_mask][cont_column]
            self.mean_score = score.mean()
            target = df[hit_mask]['targets']

            bins = np.unique(np.percentile(score, np.linspace(0,100, orig_bin_num + 1)))

            scores = []
            brs = []
            counts = []
            defs = []

            for b in zip(bins[:-1], bins[1:]):
                counts += [score[(score>=b[0]) & (score<b[1])].count()] 
                scores += [score[(score>=b[0]) & (score<b[1])].mean()]
                defs += [target[(score>=b[0]) & (score<b[1])].sum()]
                brs += [target[(score>=b[0]) & (score<b[1])].mean()]

            logit_brs = list(pd.Series(brs).apply(lambda x: np.log(x / (1-x))))    

            data = pd.DataFrame([counts, defs, scores, logit_brs]).T
            data.columns = ['Count', 'Defaults', 'Pred_MEAN', 'Target_logit_MEAN']

            data['Intercept'] = 1
            model = LinearRegression()
            model.fit(data[['Target_logit_MEAN']],data[['Pred_MEAN']], sample_weight = data['Count'])

#             predict_missing= logit(df[non_hit_mask]['targets'].mean())
#             self.imput_value_dict[cont_column] = model.predict(pd.DataFrame([predict_missing]))
            if df[non_hit_mask]['targets'].mean() in [0,1] :
                print('Pikachu:default rate of missing values in [0,1]')
                self.imput_value_dict[cont_column] = score.mean() 
                
            elif np.sum( ~df[non_hit_mask]['targets'].isnull())==0:
                print('Pikachu:no missing')
                self.imput_value_dict[cont_column] = score.mean() 
                    
            else:
                predict_missing= logit(df[non_hit_mask]['targets'].mean())
                self.imput_value_dict[cont_column] = model.predict(pd.DataFrame([predict_missing]))
                self.imput_value_dict[cont_column] = self.imput_value_dict[cont_column][0][0]
                data.drop('Intercept', axis = 1, inplace = True)
                self.predict_missing = predict_missing
                self.table = data
                self.model = model

            
        return self
    
    
    def transform(self, X):
        df = X.copy()
        if type(df) != 'pandas.core.frame.DataFrame':
            df = pd.DataFrame(df)  
        for predictor in df.columns:
            df[predictor] =  df[predictor].fillna(self.imput_value_dict[predictor])
        return df
