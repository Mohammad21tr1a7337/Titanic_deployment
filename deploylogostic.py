import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# --- UI Setup ---
st.set_page_config(page_title="🚢 Titanic Survival Predictor", layout="centered")
st.title("🚢 Titanic Survival Predictor")
st.markdown("### Would you have survived the Titanic disaster?")

# --- Sidebar Input ---
def user_input_feature():
    age = st.sidebar.slider("Age", 1, 80, 25)
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
    embarked = st.sidebar.selectbox("Embarked", ["Cherbourg", "Southampton", "Queenstown"])
    sibsp = st.sidebar.slider("Number of Siblings/Spouses", 0, 5, 0)
    parch = st.sidebar.slider("Number of Parents/Children", 0, 5, 0)

    # Convert to encoded values
    gender_code = 1 if gender == "Male" else 0
    embarked_map = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}
    embarked_code = embarked_map[embarked]

    data = {
        'Age': age,
        'Sex': gender_code,
        'Pclass': pclass,
        'SibSp': sibsp,
        'Parch': parch,
        'Embarked': embarked_code
    }

    return pd.DataFrame(data, index=[0])

data = user_input_feature()
st.markdown("#### 👤 Your Input:")
st.write(data)

# --- Load & preprocess training data ---
df = pd.read_csv("Titanic_train.csv")
df.dropna(subset=['Age', 'Embarked'], inplace=True)

# Encode categorical variables
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

# Select and scale features
features = ['Age', 'Sex', 'Pclass', 'SibSp', 'Parch']
X = df[features]
y = df['Survived']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Apply same scaling to user input
data_scaled = scaler.transform(data[features])

# --- Model ---
model = LogisticRegression(class_weight='balanced')
model.fit(X_scaled, y)
y_pred = model.predict(data_scaled)
y_prob = model.predict_proba(data_scaled)

# --- Output ---
st.markdown("### 🧠 Prediction Result")
if y_pred[0] == 1:
    st.success(f"🎉 You would have survived! (Survival probability: {y_prob[0][1]:.2f})")
else:
    st.error(f"💀 You would NOT have survived. (Survival probability: {y_prob[0][1]:.2f})")

st.markdown("#### 🔍 Probability Breakdown")
st.write({
    "Not Survived": f"{y_prob[0][0]:.2f}",
    "Survived": f"{y_prob[0][1]:.2f}"
})
