# %%
from sklearn.pipeline import Pipeline
from sklearn import ensemble 
from xgboost import XGBRegressor
import optuna

#%% 
def build_model(model='random-forest'):
    # permite alternar entre modelos na hora do treino
    rfr = ensemble.RandomForestRegressor(random_state=42)
    xgb = XGBRegressor(random_state=42)

    if model == 'random-forest':
        return rfr
    elif model == 'xgb': 
        return xgb
            
def build_params(model='random-forest'):
    # permite alternar entre hiperparametros de modelos no hora do treino
    rfr = {'model__max_depth': optuna.distributions.IntDistribution(5, 15), 
           'model__min_samples_split': optuna.distributions.IntDistribution(5, 20), 
           'model__n_estimators': optuna.distributions.IntDistribution(300, 800),
           'model__min_samples_leaf': optuna.distributions.IntDistribution(5, 25), 
           'model__ccp_alpha': optuna.distributions.FloatDistribution(0.001, 0.01)} 

    xgb = {'model__max_depth': optuna.distributions.IntDistribution(5, 15),             
           'model__min_child_weight': optuna.distributions.IntDistribution(5, 20),      
           'model__n_estimators': optuna.distributions.IntDistribution(300, 800),          
           'model__gamma': optuna.distributions.IntDistribution(500, 5000),                           
           'model__learning_rate': optuna.distributions.FloatDistribution(0.02, 0.1)}                   

    if model == 'random-forest':
        return rfr
    elif model == 'xgb': 
        return xgb
    elif model == 'teste-random-forest':
        return {'model__max_depth': optuna.distributions.IntDistribution(5, 6), 
                'model__min_samples_split': optuna.distributions.IntDistribution(5, 6), 
                'model__n_estimators': optuna.distributions.IntDistribution(2, 3)}
 