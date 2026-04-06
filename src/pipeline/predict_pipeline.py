import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")

        
            data_scaled = preprocessor.transform(features)

           
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)



class CustomData:
    def __init__(
        self,
        Age,
        Family_Income,
        Study_Hours_per_Day,
        Attendance_Rate,
        Assignment_Delay_Days,
        Travel_Time_Minutes,
        Stress_Index,
        GPA,
        Semester_GPA,
        CGPA,
        Gender,
        Internet_Access,
        Part_Time_Job,
        Scholarship,
        Semester,
        Department,
        Parental_Education
    ):

      
        self.Age = Age
        self.Family_Income = Family_Income
        self.Study_Hours_per_Day = Study_Hours_per_Day
        self.Attendance_Rate = Attendance_Rate
        self.Assignment_Delay_Days = Assignment_Delay_Days
        self.Travel_Time_Minutes = Travel_Time_Minutes
        self.Stress_Index = Stress_Index
        self.GPA = GPA
        self.Semester_GPA = Semester_GPA
        self.CGPA = CGPA

       
        self.Gender = Gender
        self.Internet_Access = Internet_Access
        self.Part_Time_Job = Part_Time_Job
        self.Scholarship = Scholarship
        self.Semester = Semester
        self.Department = Department
        self.Parental_Education = Parental_Education

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age": [self.Age],
                "Family_Income": [self.Family_Income],
                "Study_Hours_per_Day": [self.Study_Hours_per_Day],
                "Attendance_Rate": [self.Attendance_Rate],
                "Assignment_Delay_Days": [self.Assignment_Delay_Days],
                "Travel_Time_Minutes": [self.Travel_Time_Minutes],
                "Stress_Index": [self.Stress_Index],
                "GPA": [self.GPA],
                "Semester_GPA": [self.Semester_GPA],
                "CGPA": [self.CGPA],
                "Gender": [self.Gender],
                "Internet_Access": [self.Internet_Access],
                "Part_Time_Job": [self.Part_Time_Job],
                "Scholarship": [self.Scholarship],
                "Semester": [self.Semester],
                "Department": [self.Department],
                "Parental_Education": [self.Parental_Education],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
