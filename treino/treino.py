# %%
from tempfile import tempdir
import pandas as pd
import numpy as np

from sklearn import metrics, model_selection
from sklearn.pipeline import Pipeline
from optuna.integration import OptunaSearchCV 
from preprocessing import process_features
from preprocessing import process_target
from preprocessing import inverse_process
from modelo import build_model
from modelo import build_params
from metrics import metrics_report
import mlflow
import pickle

#%%
base = pd.read_csv(r'C:\dtf\CSs\DS_Academy\projetos\aplicacao_web_integracao_c_ml_-disc_1_eng_ml\dados\base.csv')
base.head()

#%%
x = base.drop(columns='Salary')
y = base['Salary']

xtrain, xtest, ytrain, ytest = model_selection.train_test_split(x, y, test_size=0.3, random_state=42)

x_cat = x.dtypes.index[x.dtypes=='object'].tolist()
x_num = x.dtypes.index[x.dtypes=='float64'].tolist()

#%%
pipe_model = Pipeline(steps=[('preprocess', process_features(x_cat=x_cat, x_num=x_num)),
                              ('model', build_model(model='random-forest'))])

params = build_params(model='random-forest')

#%%
grid_model = OptunaSearchCV(pipe_model,
                            param_distributions=params,
                            n_trials=50,
                            cv=model_selection.KFold(n_splits=5, 
                                                     shuffle=True, # evita vies embarralhando amostras
                                                     random_state=42),
                            scoring='neg_root_mean_squared_error',                                                            
                            return_train_score=True,
                            n_jobs=-1,
                            random_state=42)   

#%%
# ytrain_t: y com outliers tratados. power_t: objeto treinado com y, ultil para inversao adiante
ytrain_t, power_t = process_target(ytrain)

#%%
mlflow.set_tracking_uri(uri='http://127.0.0.1:8080')
mlflow.set_experiment(experiment_id=423294410172485549)

#%%
with mlflow.start_run():
    grid_model.fit(xtrain, ytrain_t)
    
    # previsao e inversao para escala real do y tinha sido transformado
    ytrain_pred = inverse_process(grid_model.predict(xtrain), power_t) 
    ytest_pred = inverse_process(grid_model.predict(xtest), power_t)
    
    mlflow_report = metrics_report(ytest, ytest_pred, ytrain, ytrain_pred, grid_model)
    
    mlflow.log_metrics(mlflow_report)
    mlflow.log_params(grid_model.best_params_) 
    mlflow.sklearn.log_model(grid_model, "model")

#%%
# modelo final para consumo na aplicacao web
ml_path = r'C:\dtf\CSs\DS_Academy\projetos\aplicacao_web_integracao_c_ml_-disc_1_eng_ml\modelo\ml_salary.pkl'
with open(ml_path, 'wb') as file:
    pickle.dump(grid_model, file)
