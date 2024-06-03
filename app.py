import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('diabetes_prediction_dataset.csv')

df = load_data()

# Define features and target variable
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Preprocessing pipeline
numeric_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
categorical_features = ['gender', 'smoking_history']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Append classifier to preprocessing pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression())])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline on training data
clf.fit(X_train, y_train)

# Make predictions on test data
y_pred = clf.predict(X_test)

# Calculate model accuracy
model_accuracy = accuracy_score(y_test, y_pred) * 100

# Sidebar - Input features
st.sidebar.header('User Input Features')

def user_input_features():
    age = st.sidebar.slider('Age', int(X['age'].min()), int(X['age'].max()), int(X['age'].mean()))
    bmi = st.sidebar.slider('BMI', float(X['bmi'].min()), float(X['bmi'].max()), float(X['bmi'].mean()))
    HbA1c_level = st.sidebar.slider('HbA1c Level', float(X['HbA1c_level'].min()), float(X['HbA1c_level'].max()), float(X['HbA1c_level'].mean()))
    blood_glucose_level = st.sidebar.slider('Blood Glucose Level', float(X['blood_glucose_level'].min()), float(X['blood_glucose_level'].max()), float(X['blood_glucose_level'].mean()))
    gender = st.sidebar.selectbox('Gender', X['gender'].unique())
    smoking_history = st.sidebar.selectbox('Smoking History', X['smoking_history'].unique())
    data = {'age': int(age),
            'bmi': bmi,
            'HbA1c_level': HbA1c_level,
            'blood_glucose_level': blood_glucose_level,
            'gender': gender,
            'smoking_history': smoking_history}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user input features
st.subheader('User Input features')
st.write(input_df)

# Predict
prediction = clf.predict(input_df)
st.subheader('Prediction')
prediction_result = 'The person is predicted to have diabetes.' if prediction[0] == 1 else 'The person is predicted to not have diabetes.'
st.write(prediction_result)

# Calculate prediction accuracy based on user input
if 'diabetes' in input_df.columns:
    input_df.drop('diabetes', axis=1, inplace=True)  # Remove the target column if accidentally included
prediction_accuracy = None
if 'diabetes' in df.columns:
    actual_value = df[df.drop('diabetes', axis=1).eq(input_df.iloc[0]).all(axis=1)]['diabetes'].values
    if len(actual_value) > 0:
        prediction_accuracy = 'Correct' if actual_value[0] == prediction[0] else 'Incorrect'

# Display prediction accuracy
if prediction_accuracy is not None:
    st.subheader('Prediction Accuracy')
    st.write(f'The prediction accuracy based on the input features is: {prediction_accuracy}')

# Display model accuracy
st.subheader('Model Accuracy')
st.write(f'The accuracy of the model on the test set is: {model_accuracy:.2f}%')
