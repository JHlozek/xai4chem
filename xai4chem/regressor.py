import optuna
from xgboost import XGBRegressor
from catboost import CatBoostRegressor 
from sklearn import metrics
from sklearn.model_selection import train_test_split
import shap
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import json


class Regressor:
    def __init__(self, algorithm='xgboost', n_trials=200, timeout=600):
        self.algorithm = algorithm
        self.n_trials = n_trials
        self.timeout = timeout
        self.model = None 

    def _optimize_xgboost(self, trial, X, y):
        params = {  
            'lambda': trial.suggest_int('lambda', 0, 5),
            'alpha': trial.suggest_int('alpha', 0, 5), 
            'gamma': trial.suggest_int('gamma', 0,20),
            'learning_rate': trial.suggest_float('learning_rate', 0.0001,1),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree',[0.4,0.5,0.6,0.7,0.8,1.0]),
            'colsample_bynode': trial.suggest_categorical('colsample_bynode',[0.4,0.5,0.6,0.7,0.8,1.0]),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'min_child_weight': trial.suggest_int('min_child_weight',1, 100),  
            'max_depth': trial.suggest_int('max_depth', 3, 20),  
            'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
            'random_state': trial.suggest_categorical('random_state', [0, 42]),
            'early_stopping_rounds': 10
        }


        model = XGBRegressor(**params)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, 
            y, 
            test_size=0.2, 
            random_state=42
        ) 

        model.fit(X_train,y_train,eval_set=[(X_valid,y_valid)],verbose=False)
        
        preds = model.predict(X_valid)
        r2 = metrics.r2_score(y_valid, preds)
        
        return r2

    def _optimize_catboost(self, trial, X, y):
        params = {
            'iterations':trial.suggest_int("iterations", 500, 2000),
            'learning_rate' : trial.suggest_float('learning_rate',0.0001,1),
            'reg_lambda': trial.suggest_float('reg_lambda',1, 10),
            'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
            'random_strength': trial.suggest_float('random_strength',1,10),
            'depth': trial.suggest_int('depth',5, 10),
            'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations',5,10),
            'early_stopping_rounds': 10
        }

        model = CatBoostRegressor( loss_function="RMSE",random_state=42,**params)
        X_train, X_valid, y_train, y_valid = train_test_split(
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

        y_pred = model.predict(X_valid)
        return metrics.mean_squared_error(y_valid, y_pred) 

    def train(self, X_train, y_train, default_params=True):
        if self.algorithm == 'xgboost':
            if not default_params:
                study = optuna.create_study(direction="maximize")
                study.optimize(lambda trial: self._optimize_xgboost(trial, X_train, y_train), n_trials=self.n_trials, timeout=self.timeout)
                best_params = study.best_params
                print('Best parameters for XGBoost:', best_params)
                self.model = XGBRegressor(**best_params)
            else:
                self.model = XGBRegressor()
            self.model.fit(X_train, y_train)
        elif self.algorithm == 'catboost': 
            if not default_params:
                study = optuna.create_study(direction="minimize") 
                study.optimize(lambda trial: self._optimize_catboost(trial, X_train, y_train), n_trials=self.n_trials, timeout=self.timeout)
                best_params = study.best_params
                print('Best parameters for CatBoost:', best_params)
                self.model = CatBoostRegressor(**best_params)
            else:
                self.model = CatBoostRegressor()
            self.model.fit(X_train, y_train, verbose=False)
        else:
            raise ValueError("Invalid Algorithm. Supported Algorithms: 'xgboost', 'catboost'")

    def evaluate(self, X_valid, y_valid, output_folder):
        if self.model is None:
            raise ValueError("The model has not been trained.")

        y_pred = self.model.predict(X_valid)
        joblib.dump((X_valid, y_valid, y_pred), os.path.join(output_folder, "evaluation_data.joblib"))
        
        plt.scatter(y_valid, y_pred) 
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('True vs. Predicted Values')
        plt.savefig(os.path.join(output_folder, 'evaluation_scatter_plot.png'))
        plt.close()
        
        # Metrics
        mse = metrics.mean_squared_error(y_valid, y_pred)            
        rmse = np.sqrt(mse) 
        mae = metrics.mean_absolute_error(y_valid, y_pred)
        r2 = metrics.r2_score(y_valid, y_pred)
        explained_variance = metrics.explained_variance_score(y_valid, y_pred)
        
        evaluation_metrics = {
            "Mean Squared Error": round(mse, 4),
            "Root Mean Squared Error": round(rmse, 4),
            "Mean Absolute Error": round(mae, 4),
            "R-squared Score": round(r2, 4),
            "Explained Variance Score": round(explained_variance, 4)
        } 
        
        with open(os.path.join(output_folder, 'evaluation_metrics.json'), 'w') as f:
            json.dump(evaluation_metrics, f, indent=4)
            
        return evaluation_metrics
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("The model has not been trained.")
        return self.model.predict(X)

    def explain(self, X, feature_names, output_folder):
        if self.model is None:
            raise ValueError("The model has not been trained.")
        explainer = shap.Explainer(self.model)
        explanation = explainer(X)
        
        new_explanation = shap.Explanation(
        values=explanation.values, 
        base_values=explanation.base_values, 
        data=explanation.data, 
        feature_names=feature_names
        )
        #waterfall plot
        waterfall_plot = shap.plots.waterfall(new_explanation[0], max_display=15, show=False)
        waterfall_plot.figure.savefig(os.path.join(output_folder, "interpretability_sample1.png"),  bbox_inches='tight') 
        plt.close(waterfall_plot.figure)
        
        #summary plot 
        summary_plot = shap.plots.bar(new_explanation, max_display=20,  show=False)
        summary_plot.figure.savefig(os.path.join(output_folder, "interpretability_bar_plot.png"), bbox_inches='tight') 
        plt.close(summary_plot.figure)
        
        #beeswarm plot
        beeswarm_plot = shap.plots.beeswarm(new_explanation,max_display=15, show=False)
        beeswarm_plot.figure.savefig(os.path.join(output_folder, "interpretability_beeswarm_plot.png"), bbox_inches='tight') 
        plt.close(beeswarm_plot.figure)
        
        return explanation

    def save_model(self, filename):
        if self.model is None:
            raise ValueError("The model has not been trained.")
        joblib.dump(self.model, filename)

    def load_model(self, filename):
        self.model = joblib.load(filename)