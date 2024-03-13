import optuna
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn import model_selection, metrics
import shap
import joblib

class Regressor:
    def __init__(self, selected_algorithm='xgboost'):
        self.selected_algorithm = selected_algorithm
        self.model = None 

    def _optimize_xgboost(self, trial, X, y):
        params = {  
            'lambda': trial.suggest_float('lambda', 7.0, 17.0),
            'alpha': trial.suggest_float('alpha', 7.0, 17.0),
            'eta': trial.suggest_float('eta', 0.3,1.0),
            'gamma': trial.suggest_int('gamma', 18,25),
            'learning_rate': trial.suggest_float('learning_rate', 0.0001,0.1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.3, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 400, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 8, 600),  
            'max_depth': trial.suggest_categorical('max_depth', [3, 4, 5, 6, 7]),  
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'random_state': 42
        }


        model = XGBRegressor(**params)
        X_train, X_valid, y_train, y_valid = model_selection.train_test_split(
            X, 
            y, 
            test_size=0.2, 
            random_state=42
        ) 

        model.fit(X_train,y_train,eval_set=[(X_valid,y_valid)],early_stopping_rounds=20,verbose=False)
        
        preds = model.predict(X_valid)
        r2 = metrics.r2_score(y_valid, preds)
        
        return r2

    def _optimize_catboost(self, trial, X, y):
        params = {
            'iterations':trial.suggest_int("iterations", 5000, 20000),
            'od_wait':trial.suggest_int('od_wait', 1000, 2000),
            'learning_rate' : trial.suggest_float('learning_rate',0.00001,0.1),
            'reg_lambda': trial.suggest_float('reg_lambda',1e-5,100),
            'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
            'random_strength': trial.suggest_float('random_strength',10,50),
            'depth': trial.suggest_int('depth',1, 15),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',1,20),
            'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations',1,15),
        }

        model = CatBoostRegressor( loss_function="RMSE",random_state=42,**params)
        X_train, X_valid, y_train, y_valid = model_selection.train_test_split(
            X, 
            y, 
            test_size=0.2, 
            random_state=42
        ) 

        model.fit(X_train, y_train,
            early_stopping_rounds=10,
            eval_set=[(X_valid, y_valid)], 
            verbose=False
        )

        yhat = model.predict(X_valid)
        return metrics.mean_squared_error(y_valid, yhat) 

    def train(self, X_train, y_train):
        if self.selected_algorithm == 'xgboost':
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: self._optimize_xgboost(trial, X_train, y_train), n_trials=50,  timeout=600)
            best_params = study.best_params
            print('Best parameters for XGBoost:', best_params)
            self.model = XGBRegressor(**best_params)
            self.model.fit(X_train, y_train)
        elif self.selected_algorithm == 'catboost': 
            study = optuna.create_study(direction="minimize") 
            study.optimize(lambda trial: self._optimize_catboost(trial, X_train, y_train), n_trials=50,  timeout=600)
            best_params = study.best_params
            print('Best parameters for CatBoost:', best_params)
            self.model = CatBoostRegressor(**best_params)
            self.model.fit(X_train, y_train, verbose=False)
        else:
            raise ValueError("Invalid Algorithm. Supported Algoriths: 'xgboost', 'catboost'")

    def predict(self, X):
        if self.model is None:
            raise ValueError("The model has not been trained.")
        return self.model.predict(X)

    def explain(self, X):
        if self.model is None:
            raise ValueError("The model has not been trained.")
        explainer = shap.Explainer(self.model)
        shap_values = explainer.shap_values(X)
        return shap_values

    def save_model(self, filename):
        if self.model is None:
            raise ValueError("The model has not been trained.")
        joblib.dump(self.model, filename)

    def load_model(self, filename):
        self.model = joblib.load(filename)