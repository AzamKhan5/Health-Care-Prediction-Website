from flask import Flask, render_template, request, url_for
import numpy as np, pandas as pd
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


############################################################################################################

# Heart Attack


# Load the models and scalers
model = pickle.load(open('svm_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/prediction', methods=['POST'])
def prediction():
    # Retrieve form data
    heart_features = [
        float(request.form['age']),
        float(request.form['sex']),
        float(request.form['cp']),
        float(request.form['trestbps']),
        float(request.form['chol']),
        float(request.form['fbs']),
        float(request.form['restecg']),
        float(request.form['thalach']),
        float(request.form['exang']),
        float(request.form['oldpeak']),
        float(request.form['slope']),
        float(request.form['ca']),
        float(request.form['thal']),
    ]

    # Convert input to numpy array and scale
    input_features = np.array([heart_features])
    scaled_features = scaler.transform(input_features)

    # Make prediction
    prediction_heart = model.predict(scaled_features)[0]  # No need for extra indexing

    # Interpret result
    heart_result = "Positive (Risk of Heart Attack)" if prediction_heart == 1 else "Negative (No Risk of Heart Attack)"

    return render_template('heartattack.html', prediction_heart=heart_result)
############################################################################################################


#kidney 

kidney_model = pickle.load(open('kidney_model.pkl', 'rb'))
kidney_scaler = pickle.load(open('kidney_scaler.pkl', 'rb'))

@app.route('/kidney_prediction', methods=['POST'])
def kidney_prediction():
        # Retrieve input values from form
        kidney_features = [
            float(request.form['age']),
            float(request.form['blood_pressure']),
            float(request.form['specific_gravity']),
            float(request.form['albumin']),
            float(request.form['sugar']),
            int(request.form['red_blood_cells']),
            int(request.form['pus_cell']),
            int(request.form['pus_cell_clumps']),
            int(request.form['bacteria']),
            float(request.form['blood_glucose_random']),
            float(request.form['blood_urea']),
            float(request.form['serum_creatinine']),
            float(request.form['sodium']),
            float(request.form['potassium']),
            float(request.form['haemoglobin']),
            float(request.form['packed_cell_volume']),
            float(request.form['white_blood_cell_count']),
            float(request.form['red_blood_cell_count']),
            int(request.form['hypertension']),
            int(request.form['diabetes_mellitus']),
            int(request.form['coronary_artery_disease']),
            int(request.form['appetite']),
            int(request.form['peda_edema']),
            int(request.form['anaemia']),
        ]

        # Convert to NumPy array and scale
        input_features = np.array([kidney_features])
        scaled_features = kidney_scaler.transform(input_features)

        # Make prediction
        prediction_kidney = kidney_model.predict(scaled_features)[0]

        print(f"Prediction Result: {prediction_kidney}")  # Debugging output

        # Corrected output messages
        kidney_result = "Positive (Risk of Kidney Disease)" if prediction_kidney == 1 else "Negative (No Risk of Kidney Disease)"

        return render_template('kidney.html', prediction_kidney=kidney_result)

############################################################################################################

# Lung Cancer

lung_model = pickle.load(open('lung_cancer_model.pkl', 'rb'))
lung_scaler = pickle.load(open('lung_scaler.pkl', 'rb'))

@app.route('/lung_predict', methods=['POST'])
def lung_predict():
    if request.method == 'POST':
        # Retrieve form data
        lung_features = [
            int(request.form['gender']),
            int(request.form['age']),
            int(request.form['smoking']),
            int(request.form['yellow_fingers']),
            int(request.form['anxiety']),
            int(request.form['peer_pressure']),
            int(request.form['chronic_disease']),
            int(request.form['fatigue']),
            int(request.form['allergy']),
            int(request.form['wheezing']),
            int(request.form['alcohol_consuming']),
            int(request.form['coughing']),
            int(request.form['shortness_of_breath']),
            int(request.form['swallowing_difficulty']),
            int(request.form['chest_pain'])
        ]
        
        # Convert to numpy array and reshape
        lung_input_data = np.array([lung_features]).reshape(1, -1)
        
        # Scale the input data
        lung_input_data = lung_scaler.transform(lung_input_data)
        
        # Make prediction
        lung_prediction = lung_model.predict(lung_input_data)
        
        # Map prediction output
        lung_result = "Positive for Lung Cancer" if lung_prediction[0] == 1 else "Negative for Lung Cancer"
        
        return render_template('lung_index.html', lung_prediction=lung_result)



############################################################################################################


# Breast cancer

breast_model = pickle.load(open('breast_cancer_model.pkl', 'rb'))
breast_scaler = pickle.load(open('breast_scaler.pkl', 'rb'))

feature_names = breast_model.feature_names_in_

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data
        breast_features = [
            float(request.form['radius_mean']),
            float(request.form['texture_mean']),
            float(request.form['perimeter_mean']),
            float(request.form['area_mean']),
            float(request.form['smoothness_mean']),
            float(request.form['compactness_mean']),
            float(request.form['concavity_mean']),
            float(request.form['concave_points_mean']),
            float(request.form['symmetry_mean']),
            float(request.form['fractal_dimension_mean']),
            float(request.form['radius_se']),
            float(request.form['texture_se']),
            float(request.form['perimeter_se']),
            float(request.form['area_se']),
            float(request.form['smoothness_se']),
            float(request.form['compactness_se']),
            float(request.form['concavity_se']),
            float(request.form['concave_points_se']),
            float(request.form['symmetry_se']),
            float(request.form['fractal_dimension_se']),
            float(request.form['radius_worst']),
            float(request.form['texture_worst']),
            float(request.form['perimeter_worst']),
            float(request.form['area_worst']),
            float(request.form['smoothness_worst']),
            float(request.form['compactness_worst']),
            float(request.form['concavity_worst']),
            float(request.form['concave_points_worst']),
            float(request.form['symmetry_worst']),
            float(request.form['fractal_dimension_worst'])
        ]

        # Convert to numpy array and reshape
        breast_input_data = np.array([breast_features])

        # Convert to DataFrame to match Model Feature Names
        breast_input_df = pd.DataFrame(breast_input_data, columns=feature_names)

    

        # Scale the input data
        input_data_scaled = breast_scaler.transform(breast_input_df)


        # Make prediction using DataFrame (Fix for feature name issue)
        breast_prediction = breast_model.predict(breast_input_df)


        # Map prediction output
        breast_result = "Malignant (Cancerous)" if breast_prediction[0] == 1 else "Benign (Non-Cancerous)"

        return render_template('breast_index.html', breast_prediction=breast_result)

############################################################################################################

diabetes_model = pickle.load(open('diabetes_model.pkl', 'rb'))
diabetes_scaler = pickle.load(open('diabetes_scaler.pkl', 'rb'))

@app.route('/diapredict', methods=['POST'])
def diapredict():
    if request.method == 'POST':
        try:
            # Extract user input
            diabetes_features = [
                float(request.form['Pregnancies']),
                float(request.form['Glucose']),
                float(request.form['BloodPressure']),
                float(request.form['SkinThickness']),
                float(request.form['Insulin']),
                float(request.form['BMI']),
                float(request.form['DiabetesPedigreeFunction']),
                float(request.form['Age'])
            ]
            
            # Convert to numpy array and reshape
            diabetes_input_data = np.array([diabetes_features]).reshape(1, -1)
            
            # Scale the input data
            diabetes_input_data = diabetes_scaler.transform(diabetes_input_data)
            
            # Make prediction
            diabetes_prediction = diabetes_model.predict(diabetes_input_data)
            
            # Map prediction output
            diabetes_result = "Diabetic" if diabetes_prediction[0] == 1 else "Non-Diabetic"
        
        except Exception as e:
            diabetes_result = f"Error: {e}"
        
        return render_template("diabetes_index.html", diabetes_prediction=diabetes_result)


############################################################################################################


# Load the pre-trained model and scaler
stroke_model = pickle.load(open('stroke_model.pkl', 'rb'))
stroke_scaler = pickle.load(open('brain_stroke_scaler.pkl', 'rb'))


@app.route('/stroke_predict', methods=['POST'])
def stroke_predict():
    if request.method == 'POST':
        # Retrieve form data
        gender = 1 if request.form['gender'] == 'Male' else 0
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = int(request.form['ever_married'])
        residence_type = int(request.form['residence_type'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        
        # Work Type Encoding
        work_type = request.form['work_type']
        work_type_private = 1 if work_type == 'Private' else 0
        work_type_self = 1 if work_type == 'Self-employed' else 0
        work_type_children = 1 if work_type == 'Children' else 0
        
        # Smoking Status Encoding
        smoking_status = request.form['smoking_status']
        smoking_formerly = 1 if smoking_status == 'formerly smoked' else 0
        smoking_never = 1 if smoking_status == 'never smoked' else 0
        smoking_smokes = 1 if smoking_status == 'smokes' else 0
        
        # Prepare the feature vector
        stroke_features = [
            gender, age, hypertension, heart_disease, ever_married,
            residence_type, avg_glucose_level, bmi, work_type_private,
            work_type_self, work_type_children, smoking_formerly,
            smoking_never, smoking_smokes
        ]
        
        # Convert to numpy array and reshape
        stroke_input_data = np.array([stroke_features]).reshape(1, -1)
        print("Received Input Data:", stroke_input_data)
        print("Input Shape:", stroke_input_data.shape)
        
        # Scale the input data
        stroke_input_data = stroke_scaler.transform(stroke_input_data)

        
        # Make prediction
        stroke_prediction = stroke_model.predict(stroke_input_data)
        
        # Map prediction output
        stroke_result = "Stroke Detected" if stroke_prediction[0] == 1 else "No Stroke"
        
        return render_template('brain_stroke_index.html',stroke_prediction=stroke_result)




############################################################################################################

@app.route('/aboutus')
def about_us():
    return render_template('aboutus.html')

@app.route('/contactus')
def contact_us():
    return render_template('contactus.html')

@app.route('/heartattack')
def heart_attack():
    return render_template('heartattack.html')

@app.route('/kidney')
def kidney():
    return render_template('kidney.html')


@app.route('/lung_index')
def lung_index():
    return render_template('lung_index.html')

@app.route('/breast_index')
def breast_index():
    return render_template('breast_index.html')

@app.route('/diabetes_index')
def diabetes_index():
    return render_template('diabetes_index.html')

@app.route('/stroke_index')
def stroke_index():
    return render_template('brain_stroke_index.html')

if __name__ == '__main__':
    app.run(debug=True)
