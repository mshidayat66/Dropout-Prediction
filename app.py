import joblib
import streamlit as st
import pandas as pd

# Load model, encoder, dan kolom fitur
model = joblib.load("best_model.pkl")
label_encoder = joblib.load("label_encoder_status.joblib")
model_columns = joblib.load("model_columns.pkl")

def prediction(data):
    result = model.predict(data)
    final_result = label_encoder.inverse_transform(result)[0]
    return final_result

# Nama dan logo
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://github.com/dicodingacademy/assets/raw/main/logo.png", width=130)
with col2:
    st.header('Dropout App (Prototype)')

st.subheader("Input Features")

input_dict = {}

# Label kolom yang lebih ramah
column_labels = {
    'Marital_status': 'Marital Status',
    'Course': 'Course',
    'Daytime_evening_attendance': 'Daytime Evening Attendance',
    'Previous_qualification ': 'Previous Qualification',
    'Previous_qualification_grade': 'Previous Qualification Grade',
    'Nacionality': 'Nationality',
    'Mothers_qualification': 'Mothers Qualification',
    'Fathers_qualification': 'Fathers Qualification',
    'Admission_grade': 'Admission Grade',
    'Displaced': 'Displaced',
    'Debtor': 'Debtor',
    'Tuition_fees_up_to_date': 'Tuition Fees Up to Date',
    'Gender': 'Gender',
    'Scholarship_holder': 'Scholarship Holder',
    'Age_at_enrollment ': 'Age at Enrollment',
    'International': 'International',
    'Curricular_units_1st_sem_credited': 'Curricular Units 1st Sem Credited',
    'Curricular_units_1st_sem_enrolled': 'Curricular Units 1st Sem Enrolled',
    'Curricular_units_1st_sem_evaluations': 'Curricular Units 1st Sem Evaluations',
    'Curricular_units_1st_sem_approved': 'Curricular Units 1st Sem Approved',
    'Curricular_units_1st_sem_grade ': 'Curricular Units 1st Sem Grade',
    'Curricular_units_1st_sem_without_evaluations': 'Curricular Units 1st Sem Without Evaluations',
    'Curricular_units_2nd_sem_credited': 'Curricular Units 2nd Sem Credited',
    'Curricular_units_2nd_sem_enrolled': 'Curricular Units 2nd Sem Enrolled',
    'Curricular_units_2nd_sem_evaluations': 'Curricular Units 2nd Sem Evaluations',
    'Curricular_units_2nd_sem_approved': 'Curricular Units 2nd Sem Approved',
    'Curricular_units_2nd_sem_grade ': 'Curricular Units 2nd Sem Grade',
    'Curricular_units_2nd_sem_without_evaluations': 'Curricular Units 2nd Sem Without Evaluations',
}

# Opsi yang digunakan untuk selectbox
option_mappings = {
    "Marital_status": {
        "Single": 1,
        "Married": 2,
        "Widower": 3,
        "Divorced": 4,
        "Facto Union": 5,
        "Legally Separated": 6
    },
    "Course": {
        "Biofuel Production Technologies": 33,
        "Animation and Multimedia Design": 171,
        "Social Service (evening attendance)": 8014,
        "Agronomy": 9003,
        "Communication Design": 9070,
        "Veterinary Nursing": 9085,
        "Informatics Engineering": 9119,
        "Equinculture": 9130,
        "Management": 9147,
        "Social Service": 9238,
        "Tourism": 9254,
        "Nursing": 9500,
        "Oral Hygiene": 9556,
        "Advertising and Marketing Management": 9670,
        "Journalism and Communication": 9773,
        "Basic Education": 9853,
        "Management (evening attendance)": 9991
    },
    "Daytime_evening_attendance": {
        "Evening": 0,
        "Daytime": 1
    },
    "Previous_qualification": {
        "Secondary education": 1,
        "Higher education - bachelor's degree": 2,
        "Higher education - degree": 3,
        "Higher education - master's": 4,
        "Higher education - doctorate": 5,
        "Frequency of higher education": 6,
        "12th year of schooling - not completed": 9,
        "11th year of schooling - not completed": 10,
        "Other - 11th year of schooling": 12,
        "10th year of schooling": 14,
        "10th year of schooling - not completed": 15,
        "Basic education 3rd cycle (9th/10th/11th year) or equiv": 19,
        "Basic education 2nd cycle (6th/7th/8th year) or equiv": 38,
        "Technological specialization course": 39,
        "Higher education - degree": 40,
        "Professional higher technical course": 42,
        "Higher education - master (2nd cycle)": 43
    },
    "Nacionality": {
        "Portuguese": 1,
        "German": 2,
        "Spanish": 6,
        "Italian": 11,
        "Dutch": 13,
        "English": 14,
        "Lithuanian": 17,
        "Angolan": 21,
        "Cape Verdean": 22,
        "Guinean": 24,
        "Mozambican": 25,
        "Santomean": 26,
        "Turkish": 32,
        "Brazilian": 41,
        "Romanian": 62,
        "Moldova (Republic of)": 100,
        "Mexican": 101,
        "Ukrainian": 103,
        "Russian": 105,
        "Cuban": 108,
        "Colombian": 109
    },
    "Mothers_qualification": {
        "Secondary Education - 12th Year of Schooling or Eq": 1,
        "Higher Education - Bachelor's Degree": 2,
        "Higher Education - Degree": 3,
        "Higher Education - Master's": 4,
        "Higher Education - Doctorate": 5,
        "Frequency of Higher Education": 6,
        "12th Year of Schooling - Not Completed": 9,
        "11th Year of Schooling - Not Completed": 10,
        "7th Year (Old)": 11,
        "Other - 11th Year of Schooling": 12,
        "10th Year of Schooling": 14,
        "General commerce course": 18,
        "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv": 19,
        "Technical-professional course": 22,
        "7th year of schooling": 26,
        "2nd cycle of the general high school course": 27,
        "9th Year of Schooling - Not Completed": 29,
        "8th year of schooling": 30,
        "Unknown ": 34,
        "Can't read or write": 35,
        "Can read without having a 4th year of schooling": 36,
        "Basic education 1st cycle (4th/5th year) or equiv": 37,
        "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv": 38,
        "Technological specialization course": 39,
        "Higher education - degree (1st cycle)": 40,
        "Specialized higher studies course": 41,
        "Professional higher technical course": 42,
        "Higher Education - Master (2nd cycle)": 43,
        "Higher Education - Doctorate (3rd cycle)": 44
    },
    "Fathers_qualification": {
        "Secondary Education - 12th Year of Schooling or Eq": 1,
        "Higher Education - Bachelor's Degree": 2,
        "Higher Education - Degree": 3,
        "Higher Education - Master's": 4,
        "Higher Education - Doctorate": 5,
        "Frequency of Higher Education": 6,
        "12th Year of Schooling - Not Completed": 9,
        "11th Year of Schooling - Not Completed": 10,
        "7th Year (Old)": 11,
        "Other - 11th Year of Schooling": 12,
        "2nd year complementary high school course": 13,
        "10th Year of Schooling": 14,
        "General commerce course": 18,
        "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv": 19,
        "Complementary High School Course": 20,
        "Technical-professional course": 22,
        "Complementary High School Course - not concluded": 25,
        "7th year of schooling": 26,
        "2nd cycle of the general high school course": 27,
        "9th Year of Schooling - Not Completed": 29,
        "8th year of schooling": 30,
        "General Course of Administration and Commerce": 31,
        "Supplementary Accounting and Administration": 33,
        "Unknown ": 34,
        "Can't read or write": 35,
        "Can read without having a 4th year of schooling": 36,
        "Basic education 1st cycle (4th/5th year) or equiv": 37,
        "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv": 38,
        "Technological specialization course": 39,
        "Higher education - degree (1st cycle)": 40,
        "Specialized higher studies course": 41,
        "Professional higher technical course": 42,
        "Higher Education - Master (2nd cycle)": 43,
        "Higher Education - Doctorate (3rd cycle)": 44
    },
    "Displaced": {
        "No": 0,
        "Yes": 1
    },
    "Debtor": {
        "No": 0,
        "Yes": 1
    },
    "Tuition_fees_up_to_date": {
        "No": 0,
        "Yes": 1
    },
    "Gender": {
        "Female": 0,
        "Male": 1
    },
    "Scholarship_holder": {
        "No": 0,
        "Yes": 1
    },
    "International": {
        "No": 0,
        "Yes": 1
    },
}

# Buat input berdasarkan kolom yang dibutuhkan model
for col in model_columns:
    label = column_labels.get(col, col.replace("_", " ").title())

    if col in option_mappings:
        selected_label = st.selectbox(f"{label}", list(option_mappings[col].keys()))
        input_dict[col] = option_mappings[col][selected_label]
    
    elif col in ["Previous_qualification_grade", "Admission_grade"]:
        input_dict[col] = st.number_input(f"{label}", min_value=0.0, max_value=200.00, step=0.1)

    elif col in ["Curricular_units_1st_sem_grade", "Curricular_units_2nd_sem_grade"]:
        input_dict[col] = st.number_input(f"{label}", min_value=0.0, max_value=20.0, step=0.1)

    elif col == "Age_at_enrollment":
        input_dict[col] = st.number_input(f"{label}", min_value=1, max_value=99, value=18, step=1)

    elif col in [
        "Curricular_units_1st_sem_credited",
        "Curricular_units_1st_sem_enrolled",
        "Curricular_units_1st_sem_evaluations",
        "Curricular_units_1st_sem_approved",
        "Curricular_units_1st_sem_without_evaluations",
        "Curricular_units_2nd_sem_credited",
        "Curricular_units_2nd_sem_enrolled",
        "Curricular_units_2nd_sem_evaluations",
        "Curricular_units_2nd_sem_approved",
        "Curricular_units_2nd_sem_without_evaluations",
    ]:
        input_dict[col] = st.number_input(f"{label}", min_value=0, value=0, step=1)

# Konversi input ke DataFrame
data = pd.DataFrame([input_dict])

with st.expander("Lihat Data yang Diberikan"):
    st.dataframe(data)

# Tombol Prediksi
if st.button("Predict"):
    input_df = pd.DataFrame([input_dict])
    prediction_result = prediction(input_df)
    st.success(f"Predicted Dropout Status: {prediction_result}")
