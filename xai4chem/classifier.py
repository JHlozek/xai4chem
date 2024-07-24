import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import optuna
import xgboost
import lightgbm
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from flaml.default import LGBMClassifier, XGBClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from featurewiz import FeatureWiz
from xai4chem.explain_model import explain_model

class Classifier:
    def __init__(self, output_folder, algorithm='xgboost', n_trials=500, k=None):
        self.algorithm = algorithm
        self.n_trials = n_trials
        self.output_folder = output_folder
        self.model = None
        self.max_features = k
        self.selected_features = None
        self.optimal_threshold = None  # Store the optimal threshold

    def _select_features(self, X_train, y_train):
        if self.max_features is None:
            print("No maximum feature limit specified. Using all features.")
            self.selected_features = list(X_train.columns)
        elif X_train.shape[1] <= self.max_features:
            print(f"Number of input features is less than or equal to {self.max_features}. Using all features.")
            self.selected_features = list(X_train.columns)
        else:
            print(f"Features in the dataset are more than {self.max_features}. Using Featurewiz for feature selection")
            fwiz = FeatureWiz(corr_limit=0.9, feature_engg='', category_encoders='', dask_xgboost_flag=False, nrows=None, verbose=0) 
            X_train_fwiz, _ = fwiz.fit_transform(X_train, y_train) 
            selected_features = fwiz.features
            
            if len(selected_features) >= self.max_features:
                print(f"Number of features selected by Featurewiz exceeds {self.max_features}. Selecting top {self.max_features}")
                self.selected_features = selected_features[:self.max_features]
            else: 
                print('Using Featurewiz, skipping SULO algorithm in feature selection')
                fwiz = FeatureWiz(corr_limit=0.9, skip_sulov=True, feature_engg='', category_encoders='', dask_xgboost_flag=False, nrows=None, verbose=0) 
                X_train_fwiz, _ = fwiz.fit_transform(X_train, y_train) 
                selected_features = fwiz.features
                if len(selected_features) >= self.max_features:
                    print(f"Number of features selected by Featurewiz exceeds {self.max_features}. Selecting top {self.max_features}")
                    self.selected_features = selected_features[:self.max_features]
                else:
                    print(f"Number of features selected by Featurewiz is less than {self.max_features}. Using KBest selection.")
                    selector = SelectKBest(score_func=mutual_info_classif, k=self.max_features)
                    selector.fit(X_train, y_train)
                    selected_indices = np.argsort(selector.scores_)[::-1][:self.max_features]
                    self.selected_features = X_train.columns[selected_indices]
        return self.selected_features

    def _optimize_xgboost(self, trial, X, y):
        params = {  
            'lambda': trial.suggest_int('lambda', 0, 5),
            'alpha': trial.suggest_int('alpha', 0, 5), 
            'gamma': trial.suggest_int('gamma', 0, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 1),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
            'colsample_bynode': trial.suggest_categorical('colsample_bynode', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),  
            'max_depth': trial.suggest_int('max_depth', 3, 20),  
            'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
            'random_state': trial.suggest_categorical('random_state', [0, 42]),
            'early_stopping_rounds': 10
        }

        model = XGBClassifier(**params)
        return self._train_and_evaluate_optuna_model(model, X, y)

    def _optimize_catboost(self, trial, X, y):
        params = {
            'iterations': trial.suggest_int("iterations", 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
            'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
            'random_strength': trial.suggest_float('random_strength', 1, 10),
            'depth': trial.suggest_int('depth', 5, 10),
            'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 5, 10),
            'early_stopping_rounds': 10
        }

        model = CatBoostClassifier(loss_function="Logloss", random_state=42, **params)
        return self._train_and_evaluate_optuna_model(model, X, y)

    def _train_and_evaluate_optuna_model(self, model, X, y):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42) 

        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        
        preds = model.predict(X_valid)
        accuracy = metrics.accuracy_score(y_valid, preds)
        return accuracy

    def fit(self, X_train, y_train, default_params=True):
        self._select_features(X_train, y_train)
        X_train = X_train[self.selected_features]
        
        if self.algorithm == 'xgboost':
            if not default_params:
                study = optuna.create_study(direction="maximize")
                study.optimize(lambda trial: self._optimize_xgboost(trial, X_train, y_train), n_trials=self.n_trials)
                best_params = study.best_params
                print('Best parameters for XGBoost:', best_params)
                self.model = xgboost.XGBClassifier(**best_params)
            else:
                estimator = XGBClassifier()
                (
                hyperparams,
                estimator_name,
                X_transformed,
                y_transformed,
                ) = estimator.suggest_hyperparams(X_train, y_train)
                
            self.model =  xgboost.XGBClassifier(**hyperparams)
            self.model.fit(X_train, y_train)
        elif self.algorithm == 'catboost': 
            if not default_params:
                study = optuna.create_study(direction="maximize") 
                study.optimize(lambda trial: self._optimize_catboost(trial, X_train, y_train), n_trials=self.n_trials)
                best_params = study.best_params
                print('Best parameters for CatBoost:', best_params)
                self.model = CatBoostClassifier(**best_params)
            else:
                self.model = CatBoostClassifier()
            self.model.fit(X_train, y_train, verbose=False)
        elif self.algorithm == 'lgbm': 
            estimator = LGBMClassifier()
            (
            hyperparams,
            estimator_name,
            X_transformed,
            y_transformed,
            ) = estimator.suggest_hyperparams(X_train, y_train)
                
            self.model =  lightgbm.LGBMClassifier(**hyperparams)
            self.model.fit(X_train, y_train)           
        else:
            raise ValueError("Invalid Algorithm. Supported Algorithms: xgboost, catboost")

    def evaluate(self, X_valid_features, smiles_valid, y_valid):
        if self.model is None:
            raise ValueError("The model has not been trained.")

        y_proba = self.model.predict_proba(X_valid_features)[:, 1]

        # Calculate ROC curve and optimal thresholds
        fpr, tpr, thresholds = metrics.roc_curve(y_valid, y_proba)
        roc_auc = metrics.auc(fpr, tpr) 

        self.optimal_threshold = thresholds[np.argmax(tpr - fpr)] # Youden's J statistic
        optimal_idx = np.argmax(tpr - fpr)
        optimal_tpr = tpr[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        print(f"Optimal Threshold (Youden's J): {round(self.optimal_threshold, 4)}")
        print(f"Corresponding TPR: {round(optimal_tpr, 4)}")
        print(f"Corresponding FPR: {round(optimal_fpr, 4)}")

        optimal_threshold_fpr_5 = thresholds[fpr <= 0.05][-1] # 0.05 is max_fpr
        print(f"Optimal Threshold (Max FPR 0.05): {round(optimal_threshold_fpr_5,4)}")
        optimal_threshold_fpr_10 = thresholds[fpr <= 0.1][-1] # 0.1 is max_fpr
        print(f"Optimal Threshold (Max FPR 0.1): {round(optimal_threshold_fpr_10, 4)}")

        # Predictions using default threshold (0.5)
        y_pred_default = (y_proba >= 0.5).astype(int)

        # Predictions using optimal threshold (Youden's J)
        y_pred_optimal = (y_proba >= self.optimal_threshold).astype(int)

        # Predictions using optimal threshold (Max FPR 0.05)
        y_pred_fpr_5 = (y_proba >= optimal_threshold_fpr_5).astype(int)
        # Predictions using optimal threshold (Max FPR 0.1)
        y_pred_fpr_10 = (y_proba >= optimal_threshold_fpr_10).astype(int)

        # Save results for all thresholds
        evaluation_data = pd.DataFrame({
            'SMILES': smiles_valid,
            'Actual Value': y_valid,            
            'Prob_Pred': y_proba,
            'Predicted Value (Default Threshold)': y_pred_default,
            'Predicted Value (Optimal Threshold - Youden\'s J)': y_pred_optimal,
            'Predicted Value (Optimal Threshold - Max FPR 0.05)': y_pred_fpr_5,
            'Predicted Value (Optimal Threshold - Max FPR 0.1)': y_pred_fpr_10
        })

        evaluation_data.to_csv(os.path.join(self.output_folder, "evaluation_data.csv"), index=False)

        # Confusion Matrix for all thresholds
        cm_display_default = ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(y_valid, y_pred_default))
        cm_display_default.plot() 
        cm_display_default.ax_.set_title("Confusion Matrix - Default Threshold")
        cm_display_default.figure_.savefig(os.path.join(self.output_folder, 'confusion_matrix_default.png'))
        plt.close(cm_display_default.figure_)

        cm_display_optimal = ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(y_valid, y_pred_optimal))
        cm_display_optimal.plot() 
        cm_display_optimal.ax_.set_title(f"Confusion Matrix - Optimal Threshold (Youden's J({round(self.optimal_threshold, 4)}))")
        cm_display_optimal.figure_.savefig(os.path.join(self.output_folder, 'confusion_matrix_optimal.png'))
        plt.close(cm_display_optimal.figure_)

        cm_display_fpr_5 = ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(y_valid, y_pred_fpr_5))
        cm_display_fpr_5.plot() 
        cm_display_fpr_5.ax_.set_title(f"Confusion Matrix - Optimal Threshold (Max FPR_0.05({round(optimal_threshold_fpr_5,4)}))")
        cm_display_fpr_5.figure_.savefig(os.path.join(self.output_folder, 'confusion_matrix_fpr_5%.png'))
        plt.close(cm_display_fpr_5.figure_)

        cm_display_fpr_10 = ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(y_valid, y_pred_fpr_10))
        cm_display_fpr_10.plot() 
        cm_display_fpr_10.ax_.set_title(f"Confusion Matrix - Optimal Threshold (Max FPR_0.1({round(optimal_threshold_fpr_10, 4)}))")
        cm_display_fpr_10.figure_.savefig(os.path.join(self.output_folder, 'confusion_matrix_fpr_10%.png'))
        plt.close(cm_display_fpr_10.figure_)

        # ROC Curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.scatter([optimal_fpr], [optimal_tpr], color='red', label=f'Youden Index({round(optimal_fpr, 4)}, {round(optimal_tpr, 4)})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.output_folder, 'roc_curve.png'))
        plt.close()

        # Other evaluation metrics for all thresholds
        metrics_default = {
            "Accuracy": round(metrics.accuracy_score(y_valid, y_pred_default), 4),
            "Precision": round(metrics.precision_score(y_valid, y_pred_default, average='macro'), 4),
            "Recall": round(metrics.recall_score(y_valid, y_pred_default, average='macro'), 4),
            "F1 Score": round(metrics.f1_score(y_valid, y_pred_default, average='macro'), 4)
        }

        metrics_optimal = {
            "Accuracy": round(metrics.accuracy_score(y_valid, y_pred_optimal), 4),
            "Precision": round(metrics.precision_score(y_valid, y_pred_optimal, average='macro'), 4),
            "Recall": round(metrics.recall_score(y_valid, y_pred_optimal, average='macro'), 4),
            "F1 Score": round(metrics.f1_score(y_valid, y_pred_optimal, average='macro'), 4)
        }

        metrics_fpr_5 = {
            "Accuracy": round(metrics.accuracy_score(y_valid, y_pred_fpr_5), 4),
            "Precision": round(metrics.precision_score(y_valid, y_pred_fpr_5, average='macro'), 4),
            "Recall": round(metrics.recall_score(y_valid, y_pred_fpr_5, average='macro'), 4),
            "F1 Score": round(metrics.f1_score(y_valid, y_pred_fpr_5, average='macro'), 4)
        }

        metrics_fpr_10 = {
            "Accuracy": round(metrics.accuracy_score(y_valid, y_pred_fpr_10), 4),
            "Precision": round(metrics.precision_score(y_valid, y_pred_fpr_10, average='macro'), 4),
            "Recall": round(metrics.recall_score(y_valid, y_pred_fpr_10, average='macro'), 4),
            "F1 Score": round(metrics.f1_score(y_valid, y_pred_fpr_10, average='macro'), 4)
        }

        with open(os.path.join(self.output_folder, 'evaluation_metrics_default.json'), 'w') as f:
            json.dump(metrics_default, f, indent=4)

        with open(os.path.join(self.output_folder, 'evaluation_metrics_optimal.json'), 'w') as f:
            json.dump(metrics_optimal, f, indent=4)

        with open(os.path.join(self.output_folder, 'evaluation_metrics_fpr_5.json'), 'w') as f:
            json.dump(metrics_fpr_5, f, indent=4)

        with open(os.path.join(self.output_folder, 'evaluation_metrics_fpr_10.json'), 'w') as f:
            json.dump(metrics_fpr_10, f, indent=4)    

        print("Evaluation metrics (Default Threshold):", metrics_default)        
        print(classification_report(y_valid, y_pred_default))
        print("Evaluation metrics (Optimal Threshold - Youden's J):", metrics_optimal)
        print(classification_report(y_valid, y_pred_optimal))
        print("Evaluation metrics (Optimal Threshold - Max FPR 0.05):", metrics_fpr_5)
        print(classification_report(y_valid, y_pred_fpr_5))
        print("Evaluation metrics (Optimal Threshold - Max FPR 0.1):", metrics_fpr_10)
        print(classification_report(y_valid, y_pred_fpr_10))
        return metrics_default, metrics_optimal, metrics_fpr_5, metrics_fpr_10


    def predict(self, X):
        if self.model is None:
            raise ValueError("The model has not been trained.")
        X = X[self.selected_features]
        y_proba = self.model.predict_proba(X)[:, 1]
        y_pred_default = (y_proba >= 0.5).astype(int)
        y_pred_optimal = (y_proba >= self.optimal_threshold).astype(int)
        return y_pred_default, y_pred_optimal

    def explain(self, X_features, smiles_list=None, use_fingerprints=False):
        if self.model is None:
            raise ValueError("The model has not been trained.")
        X = X_features[self.selected_features]
        explanation = explain_model(self.model, X, smiles_list, use_fingerprints, self.output_folder)
        return explanation

    def save_model(self, filename):
        if self.model is None:
            raise ValueError("The model has not been trained.")
        joblib.dump(self.model, filename)

    def load_model(self, filename):
        self.model = joblib.load(filename)