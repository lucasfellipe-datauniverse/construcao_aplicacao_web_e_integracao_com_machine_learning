#%%
import pandas as pd
import numpy as np
from sklearn import metrics

#%%
def metrics_report(ytrue_test, ypred_test, ytrue_train, ypred_train, model):
        # retorna um relatorio com metricas de qualidade     
        df_cv_results = pd.DataFrame(model.cv_results_)
        best_index = model.best_index_
        report = {
        'rmse_train': metrics.root_mean_squared_error(ytrue_train, ypred_train),        
        'rmse_test': metrics.root_mean_squared_error(ytrue_test, ypred_test),
        
        'r2_train': metrics.r2_score(ytrue_train, ypred_train),
        'r2_test': metrics.r2_score(ytrue_test, ypred_test),
        
        'nrmse_train': metrics.root_mean_squared_error(ytrue_train, ypred_train) / np.mean(ytrue_train),
        'nrmse_test': metrics.root_mean_squared_error(ytrue_test, ypred_test) / np.mean(ytrue_test),

        # sinal negativo em df_cv_results para o valor voltar ao normal(optuna retorna score negativo) 
        # aqui os valores do cv servirao apenas para avaliar estabilidade do modelo, pois os valores estao em escalas
        # transformadas devido ao tratamento de outliers que é feito no y                                
        'rmse_train_cv': -df_cv_results.loc[best_index, 'mean_train_score'], 
        'rmse_test_cv': -df_cv_results.loc[best_index, 'mean_test_score'],
        
        'std_train_cv': df_cv_results.loc[best_index, 'std_train_score'],
        'std_test_cv': df_cv_results.loc[best_index, 'std_test_score']
        }
        return report
         