from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__, template_folder='templates')
app = application


# 🏠 Home Route
@app.route('/')
def index():
    return render_template('index.html')


# 🔮 Prediction Route
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    else:
        try:
            # ✅ Collect data from form
            data = CustomData(
                Age=float(request.form.get('Age')),
                Family_Income=float(request.form.get('Family_Income')),
                Study_Hours_per_Day=float(request.form.get('Study_Hours_per_Day')),
                Attendance_Rate=float(request.form.get('Attendance_Rate')),
                Assignment_Delay_Days=float(request.form.get('Assignment_Delay_Days')),
                Travel_Time_Minutes=float(request.form.get('Travel_Time_Minutes')),
                Stress_Index=float(request.form.get('Stress_Index')),
                GPA=float(request.form.get('GPA')),
                Semester_GPA=float(request.form.get('Semester_GPA')),
                CGPA=float(request.form.get('CGPA')),
                Gender=request.form.get('Gender'),
                Internet_Access=request.form.get('Internet_Access'),
                Part_Time_Job=request.form.get('Part_Time_Job'),
                Scholarship=request.form.get('Scholarship'),
                Semester=float(request.form.get('Semester')),
                Department=request.form.get('Department'),
                Parental_Education=request.form.get('Parental_Education')
            )

            # ✅ Convert to DataFrame
            pred_df = data.get_data_as_data_frame()
            print("Input DataFrame:\n", pred_df)

            # ✅ Prediction pipeline
            predict_pipeline = PredictPipeline()

            print("Before Prediction")
            results = predict_pipeline.predict(pred_df)
            print("After Prediction")

            # ✅ Convert result to readable output
            final_result = "Dropout" if results[0] == 1 else "No Dropout"

            return render_template('home.html', results=final_result)

        except Exception as e:
            return f"Error occurred: {str(e)}"


# 🚀 Run App
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)