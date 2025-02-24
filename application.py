from flask import Flask, request, render_template, jsonify
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        try:
            data = CustomData(
                LIMIT_BAL=float(request.form.get('LIMIT_BAL', 0)),
                SEX=int(request.form.get('SEX', 0)),
                EDUCATION=int(request.form.get('EDUCATION', 0)),
                MARRIAGE=int(request.form.get('MARRIAGE', 0)),
                AGE=int(request.form.get('AGE', 0)),
                PAY_0=int(request.form.get('PAY_0', 0)),
                PAY_2=int(request.form.get('PAY_2', 0)),
                PAY_3=int(request.form.get('PAY_3', 0)),
                PAY_4=int(request.form.get('PAY_4', 0)),
                PAY_5=int(request.form.get('PAY_5', 0)),
                PAY_6=int(request.form.get('PAY_6', 0)),
                BILL_AMT1=float(request.form.get('BILL_AMT1', 0)),
                BILL_AMT2=float(request.form.get('BILL_AMT2', 0)),
                BILL_AMT3=float(request.form.get('BILL_AMT3', 0)),
                BILL_AMT4=float(request.form.get('BILL_AMT4', 0)),
                BILL_AMT5=float(request.form.get('BILL_AMT5', 0)),
                BILL_AMT6=float(request.form.get('BILL_AMT6', 0)),
                PAY_AMT1=float(request.form.get('PAY_AMT1', 0)),
                PAY_AMT2=float(request.form.get('PAY_AMT2', 0)),
                PAY_AMT3=float(request.form.get('PAY_AMT3', 0)),
                PAY_AMT4=float(request.form.get('PAY_AMT4', 0)),
                PAY_AMT5=float(request.form.get('PAY_AMT5', 0)),
                PAY_AMT6=float(request.form.get('PAY_AMT6', 0))
            )

            final_new_data = data.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(final_new_data)

            result = "Default" if pred[0] == 1 else "No Default"

            return render_template('form.html', final_result=result)

        except Exception as e:
            return render_template('form.html', final_result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
