import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data (insert your dataset path)
file_path ="C:\\Users\\user\\Downloads\\INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.csv"
data = pd.read_csv(file_path)

# Streamlit app
st.title("Employee Performance Rating Predictor")

#create a description paragraph
st.write(''' INX human resource policies are considered as employee friendly and widely perceived as best
practices in the industry.Recent years, the employee performance indexes are not healthy and this is becoming a growing
concerns among the top management. There has been increased escalations on service delivery and
client satisfaction levels came down by 8 percentage points.The model objective is to predict employee performance to assist the company in
          making more accurate hiring decisions and automating the process.
''')

num_rows = st.slider("Select the number of rows", min_value = 1, max_value = len(data), value = 5)
st.write("Here are the rows you have selected in the Dataset")
st.write(data.head(num_rows)) #st.write is the print function in python
st.write('The number of rows and columns in the dataset')
st.write(data.shape)
st.write("number of duplicates:", data[data.duplicated()])

# Drop columns that won't be used for prediction
columns_to_drop = ['EmpNumber', 'Attrition', 'PerformanceRating','Age','MaritalStatus','Gender',
                   'TrainingTimesLastYear','DistanceFromHome',]
X = data.drop(columns=columns_to_drop)
y = data['PerformanceRating']

# Encode categorical features and save class labels
label_encoders = {}
class_labels = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le
    class_labels[column] = le.classes_  # Save class labels

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X, y)



# Function to encode user inputs using the same encoders as the training data
def encode_inputs(user_inputs, label_encoders):
    encoded_inputs = {}
    for feature, value in user_inputs.items():
        if feature in label_encoders:
            encoder = label_encoders[feature]
            encoded_inputs[feature] = encoder.transform([value])[0]
        else:
            encoded_inputs[feature] = value
    return np.array(list(encoded_inputs.values())).reshape(1, -1)


st.title("USER INPUT")

# Collect user input
user_inputs = {}
for column in X.columns:
    if column in class_labels:
        user_inputs[column] = st.selectbox(column, options=class_labels[column])
    else:
        user_inputs[column] = st.number_input(column, min_value=float(X[column].min()), max_value=float(X[column].max()), value=float(X[column].mean()))

# Predict button
if st.button('Predict Performance Rating'):
    encoded_inputs = encode_inputs(user_inputs, label_encoders)
    prediction = model.predict(encoded_inputs)
    st.write(f'The predicted Performance Rating is: {int(prediction[0])}')