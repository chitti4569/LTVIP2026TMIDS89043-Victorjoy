from flask import Flask, render_template, request
import pandas as pd
import joblib
app = Flask(__name__)
model = joblib.load("floods.save")
@app.route('/')
def home():
    return render_template("home.html")
@app.route('/predict_page')
def predict_page():
    return render_template("predict.html")
@app.route('/predict', methods=['POST'])
def predict():
    try:
        cloud = float(request.form["cloud_cover"])
        annual = float(request.form["annual"])
        jan_feb = float(request.form["jan_feb"])
        mar_may = float(request.form["mar_may"])
        jun_sep = float(request.form["jun_sep"])

        input_data = pd.DataFrame([{
            "Cloud Cover": cloud,
            "ANNUAL": annual,
            "Jan-Feb": jan_feb,
            "Mar-May": mar_may,
            "Jun-Sep": jun_sep
        }])
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            result = "Possible of Severe Flood"
            color = "red"
        else:
            result = "Not possible of Severe Flood"
            color = "green"
        return render_template("result.html",
                               prediction=result,
                               color=color)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
