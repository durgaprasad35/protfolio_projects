import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class Transformationconfig:
    preprocessing_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = Transformationconfig()

    def get_data_transformer_object(self):
        try:
            # ✅ Removed Student_ID (important improvement)
            numerical_cols = [
                'Age', 'Family_Income', 'Study_Hours_per_Day',
                'Attendance_Rate', 'Assignment_Delay_Days',
                'Travel_Time_Minutes', 'Stress_Index',
                'GPA', 'Semester_GPA', 'CGPA'
            ]

            categorical_col = [
                'Gender', 'Internet_Access', 'Part_Time_Job',
                'Scholarship', 'Semester', 'Department',
                'Parental_Education'
            ]

            # ✅ Numerical Pipeline
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # ✅ Categorical Pipeline (FIXED ❗ removed scaler)
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            # ✅ Column Transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_cols),
                    ('cat', cat_pipeline, categorical_col)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Dropout"

            # ✅ Fixed drop (no axis error)
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # ✅ Combine features + target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessing_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessing_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)