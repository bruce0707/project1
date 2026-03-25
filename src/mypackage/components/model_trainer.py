import os
import sys
from mypackage.logger import logging
from mypackage.exception import CustomException
from dataclasses import dataclass
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,)
from mypackage.utils import save_object,evaluate_models
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from mypackage.exception import CustomException
from mypackage.logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_trainer(self, train_array, test_array):
         try:
                logging.info("Model Trainer initiated.")

                X_train, y_train = train_array[:, :-1], train_array[:, -1]
                X_test, y_test = test_array[:, :-1], test_array[:, -1]

                models = {
                    "Random Forest": RandomForestRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "AdaBoost": AdaBoostRegressor(),
                    "CatBoost": CatBoostRegressor(verbose=False),
                    "XGBoost": XGBRegressor(),
                    "KNN": KNeighborsRegressor(),
                    "Decision Tree": DecisionTreeRegressor()
                }
                params = {
                                "Random Forest": {
                                    'n_estimators': [100, 200],
                                    'max_depth': [10, 20],           
                                    'min_samples_split': [2, 5],
                                    'min_samples_leaf': [1, 2],
                                    'max_features': ['sqrt', 'log2']  
                                },
                                
                                "Gradient Boosting": {
                                    'n_estimators': [100, 200],
                                    'learning_rate': [0.01, 0.1],
                                    'max_depth': [3, 5],             
                                    'subsample': [0.8, 0.9]
                                },
                                
                                "AdaBoost": {
                                    'n_estimators': [50, 100],
                                    'learning_rate': [0.1, 1.0]
                                },
                                
                                "CatBoost": {
                                    'iterations': [100, 200],
                                    'learning_rate': [0.01, 0.1],
                                    'depth': [4, 6]
                                },
                                
                                "XGBoost": {
                                    'n_estimators': [100, 200],
                                    'learning_rate': [0.01, 0.1],
                                    'max_depth': [3, 6],
                                    'subsample': [0.8, 0.9]
                                },
                                
                                "KNN": {
                                    'n_neighbors': [3, 5, 7],
                                    'weights': ['uniform', 'distance']
                                },
                                
                                "Decision Tree": {
                                    'max_depth': [5, 10],            
                                    'min_samples_split': [2, 5],
                                    'min_samples_leaf': [1, 2],
                                    'max_features': ['sqrt', 'log2'] 
                                }
                            }



                model_report = evaluate_models(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    models=models,
                    params=params
                )

                best_model_score = max(model_report.values())

                best_model_name = list(model_report.keys())[ 
                    list(model_report.values()).index(best_model_score)
                ]

                best_model = models[best_model_name]

                # Train the best model
                best_model.fit(X_train, y_train)

                logging.info(f"Best model: {best_model_name}, R2 score: {best_model_score}")

                if best_model_score < 0.6:
                    raise CustomException("No good model found", sys)

                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
                )

                predicted = best_model.predict(X_test)

                r2_square = r2_score(y_test, predicted)

                print("Final R2 Score:", r2_square)
                logging.info(f"Final R2 Score: {r2_square}")

                return r2_square

         except Exception as e:
                raise CustomException(e, sys)
         
if __name__ == "__main__":

    from mypackage.components.data_transformation import DataTransformation
    from mypackage.components.data_ingestion import DataIngestion

    # Step 1: Data Ingestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # Step 2: Data Transformation
    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation(
        train_path,
        test_path
    )

    # Step 3: Model Training
    trainer = ModelTrainer()
    r2 = trainer.initiate_model_trainer(train_arr, test_arr)

    print("Final Model R2 Score:", r2)