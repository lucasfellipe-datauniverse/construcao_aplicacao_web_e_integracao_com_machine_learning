#%%
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from category_encoders import TargetEncoder

#%%
def process_features(x_cat, x_num):
    # faz transformacoes necessarias nas features como encoding e tratamento de missing
    pipe_cat = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                            ('mean_encoder', TargetEncoder(smoothing=20))])

    pipe_num = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))])

    preprocess = ColumnTransformer([('cat_transform', pipe_cat, x_cat),
                                    ('num_transform', pipe_num, x_num)])
    
    return preprocess

def process_target(y_train):
    # trata o y pois os o random forest é sensivel a eles
    power_t = PowerTransformer(method='yeo-johnson')
    y_transform = power_t.fit_transform(y_train.to_frame()).ravel() # fit requer frame, sklearn exigirá array     
    return y_transform, power_t

def inverse_process(y_pred_transformed, power_t):
    # retorna o y previsto ao normal para interpretacao             
    y_normal = power_t.inverse_transform(y_pred_transformed.reshape(-1, 1)).ravel()
    return y_normal


