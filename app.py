# import os
# import pandas as pd
# import numpy as np
# from flask import Flask, render_template, request
# from sklearn.preprocessing import StandardScaler
# import pickle
#
# app = Flask(__name__)
#
# # Global variables to store model and preprocessing objects
# model = None
# scaler_X = None
# feature_columns = None
#
# # Load pre-trained model and scaler
# def load_model_and_scaler():
#     global model, scaler_X, feature_columns
#     model_filename = 'svr_model.sav'
#     scaler_filename = 'scaler_X.pkl'
#
#     # Load model
#     with open(model_filename, 'rb') as model_file:
#         model = pickle.load(model_file)
#
#     # Load scaler
#     with open(scaler_filename, 'rb') as scaler_file:
#         scaler_X = pickle.load(scaler_file)
#
#     # Specify feature columns used in training
#     feature_columns = ['Age', 'Duration of Internship (months)', 'Performance Score', 'Attendance Rate', 'Number of Completed Projects', 'Technical Skill Rating', 'Soft Skill Rating', 'Hours Worked per Week', 'Distance from Work (miles)', 'Recommendation Score', 'Department_Data Science', 'Department_Finance', 'Department_Human Resources', 'Department_Marketing', 'Department_Web Development', 'Socioeconomic Status_High', 'Socioeconomic Status_Low', 'Socioeconomic Status_Medium', 'Mentorship Level_High', 'Mentorship Level_Low', 'Mentorship Level_Medium']
#
# # Preprocess data for predictions
# def preprocess_data(data):
#     # Drop specific columns
#     columns_to_drop = ['Intern Name', 'Skill Composite Score',
#                        'Performance-Project Interaction',
#                        'Placement Status',
#                        'Socioeconomic-Engagement Interaction']
#     df = data.drop(columns=columns_to_drop, axis=1)
#
#     # One-hot encode categorical columns
#     categorical_columns = ['Department', 'Socioeconomic Status', 'Mentorship Level']
#     df_encoded = pd.get_dummies(df,
#                                  columns=categorical_columns,
#                                  prefix=categorical_columns)
#
#     # Ensure all expected columns are present
#     for col in feature_columns:
#         if col not in df_encoded.columns:
#             df_encoded[col] = 0
#
#     # Reorder columns
#     df_encoded = df_encoded[feature_columns]
#
#     return df_encoded
#
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     global model, scaler_X, feature_columns
#
#     data = None
#     prediction_result = None
#
#     if request.method == 'POST':
#         if 'predict' in request.form:
#             if model is not None:
#                 user_input = {feature: request.form.get(feature, 0) for feature in feature_columns}
#                 input_df = pd.DataFrame([user_input])
#                 input_scaled = scaler_X.transform(input_df)
#                 prediction = model.predict(input_scaled)[0]
#                 prediction_result = round(prediction, 2)
#
#     return render_template('index.html',
#                            data=data,
#                            prediction_result=prediction_result,
#                            feature_columns=feature_columns)
#
#
# if __name__ == '__main__':
#     load_model_and_scaler()
#     app.run(debug=True)


#
#
# import os
# import pandas as pd
# import numpy as np
# from flask import Flask, render_template, request, jsonify
# from sklearn.preprocessing import StandardScaler
# import pickle
#
# app = Flask(__name__)
#
# # Global variables to store model and preprocessing objects
# model = None
# scaler_X = None
# feature_columns = None
#
#
# # Load pre-trained model and scaler
# def load_model_and_scaler():
#     global model, scaler_X, feature_columns
#     model_filename = 'svr_model.sav'
#     scaler_filename = 'scaler_X.pkl'
#
#     # Load model
#     with open(model_filename, 'rb') as model_file:
#         model = pickle.load(model_file)
#
#     # Load scaler
#     with open(scaler_filename, 'rb') as scaler_file:
#         scaler_X = pickle.load(scaler_file)
#
#     # Specify feature columns used in training
#     feature_columns = [
#         'Age', 'Duration of Internship (months)', 'Performance Score',
#         'Attendance Rate', 'Number of Completed Projects',
#         'Technical Skill Rating', 'Soft Skill Rating',
#         'Hours Worked per Week', 'Distance from Work (miles)',
#         'Recommendation Score', 'Department_Data Science',
#         'Department_Finance', 'Department_Human Resources',
#         'Department_Marketing', 'Department_Web Development',
#         'Socioeconomic Status_High', 'Socioeconomic Status_Low',
#         'Socioeconomic Status_Medium', 'Mentorship Level_High',
#         'Mentorship Level_Low', 'Mentorship Level_Medium'
#     ]
#
#
# # Route for home page
# @app.route('/')
# def home():
#     return render_template('home.html')
#
#
# # Route for intern placement prediction
# @app.route('/intern-placement', methods=['GET', 'POST'])
# def index():
#     global model, scaler_X, feature_columns
#     data = None
#     prediction_result = None
#
#     if request.method == 'POST':
#         if 'predict' in request.form:
#             if model is not None:
#                 # Collect input features
#                 user_input = {feature: float(request.form.get(feature, 0)) for feature in feature_columns}
#                 input_df = pd.DataFrame([user_input])
#
#                 # Scale input
#                 input_scaled = scaler_X.transform(input_df)
#
#                 # Make prediction
#                 prediction = model.predict(input_scaled)[0]
#                 prediction_result = round(prediction, 2)
#
#     return render_template('index.html',
#                            data=data,
#                            prediction_result=prediction_result,
#                            feature_columns=feature_columns)
#
#
# # Route for lead conversion prediction (placeholder)
# @app.route('/lead-conversion', methods=['GET', 'POST'])
# def lead_conversion():
#     prediction_result = None
#
#     if request.method == 'POST':
#         # Placeholder for lead conversion prediction logic
#         # You'll need to replace this with your actual lead conversion model
#         age = float(request.form.get('age', 0))
#         income = float(request.form.get('income', 0))
#
#         # Example simple prediction logic (replace with your actual model)
#         prediction = (age * 0.5 + income * 0.01) / 100
#         prediction_result = round(prediction * 100, 2)
#
#     return render_template('lead_conversion.html', prediction_result=prediction_result)
#
#
# if __name__ == '__main__':
#     load_model_and_scaler()
#     app.run(debug=True)

#
#
# import os
# import pandas as pd
# import numpy as np
# from flask import Flask, render_template, request, jsonify
# import pickle
#
# app = Flask(__name__)
#
# # Global variables to store models and preprocessing objects
# intern_model = None
# intern_scaler = None
# intern_feature_columns = None
#
# lead_model = None
# lead_scaler = None
# lead_feature_columns = None
#
# # Load pre-trained models and scalers
# def load_models_and_scalers():
#     global intern_model, intern_scaler, intern_feature_columns
#     global lead_model, lead_scaler, lead_feature_columns
#
#     # Intern placement model
#     with open('svr_model.sav', 'rb') as model_file:
#         intern_model = pickle.load(model_file)
#
#     with open('scaler_X.pkl', 'rb') as scaler_file:
#         intern_scaler = pickle.load(scaler_file)
#
#     intern_feature_columns = [
#         'Age', 'Duration of Internship (months)', 'Performance Score',
#         'Attendance Rate', 'Number of Completed Projects',
#         'Technical Skill Rating', 'Soft Skill Rating',
#         'Hours Worked per Week', 'Distance from Work (miles)',
#         'Recommendation Score', 'Department_Data Science',
#         'Department_Finance', 'Department_Human Resources',
#         'Department_Marketing', 'Department_Web Development',
#         'Socioeconomic Status_High', 'Socioeconomic Status_Low',
#         'Socioeconomic Status_Medium', 'Mentorship Level_High',
#         'Mentorship Level_Low', 'Mentorship Level_Medium'
#     ]
#
#     # Lead conversion model
#     with open('gb_chi_model.sav', 'rb') as model_file:
#         lead_model = pickle.load(model_file)
#
#     # If a scaler is used for lead conversion, load it here
#     # Uncomment and adjust the line below if a scaler is saved
#     # with open('scaler_lead.pkl', 'rb') as scaler_file:
#     #     lead_scaler = pickle.load(scaler_file)
#
#     # Define feature columns for lead conversion
#     lead_feature_columns = [
#     'Potential Score',
#     'Interest vs Lead Quality',
#     'Interaction Time-Hour',
#     'Lead Interest Level',
#     'Time Spent on Website',
#     'Log Time Spent on Website',
#     'Course Fee Offered',
#     'Engagement Score',
#     'Days Since Last Interaction',
#     'Total Visits'
# ]
#
#
# @app.route('/')
# def home():
#     return render_template('home.html')
#
# @app.route('/intern-placement', methods=['GET', 'POST'])
# def intern_placement():
#     global intern_model, intern_scaler, intern_feature_columns
#     prediction_result = None
#
#     if request.method == 'POST' and 'predict' in request.form:
#         if intern_model is not None:
#             user_input = {feature: float(request.form.get(feature, 0)) for feature in intern_feature_columns}
#             input_df = pd.DataFrame([user_input])
#             input_scaled = intern_scaler.transform(input_df)
#             prediction = intern_model.predict(input_scaled)[0]
#             prediction_result = round(prediction, 2)
#
#     return render_template('index.html',
#                            prediction_result=prediction_result,
#                            feature_columns=intern_feature_columns)
#
# @app.route('/lead-conversion', methods=['GET', 'POST'])
# def lead_conversion():
#     global lead_model, lead_scaler, lead_feature_columns
#     prediction_result = None
#
#     if request.method == 'POST' and 'predict' in request.form:
#         if lead_model is not None:
#             user_input = {feature: float(request.form.get(feature, 0)) for feature in lead_feature_columns}
#             input_df = pd.DataFrame([user_input])
#
#             # If scaler is used, apply scaling
#             if lead_scaler:
#                 input_scaled = lead_scaler.transform(input_df)
#             else:
#                 input_scaled = input_df
#
#             prediction = lead_model.predict(input_scaled)[0]
#             prediction_result = "Converted" if prediction == 1 else "Not Converted"
#
#     return render_template('lead_conversion.html',
#                            prediction_result=prediction_result,
#                            feature_columns=lead_feature_columns)
#
# if __name__ == '__main__':
#     load_models_and_scalers()
#     app.run(debug=True)

import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_mail import Mail, Message
import pickle

app = Flask(__name__)

# Email configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'serahrhys@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'fykh zpog atqa xaba'  # Replace with your app password
mail = Mail(app)

# Global variables to store models and preprocessing objects
intern_model = None
intern_scaler = None
intern_feature_columns = None
lead_model = None
lead_scaler = None
lead_feature_columns = None


def load_models_and_scalers():
    global intern_model, intern_scaler, intern_feature_columns
    global lead_model, lead_scaler, lead_feature_columns

    # Intern placement model loading
    with open('svr_model.sav', 'rb') as model_file:
        intern_model = pickle.load(model_file)

    with open('scaler_X.pkl', 'rb') as scaler_file:
        intern_scaler = pickle.load(scaler_file)

    intern_feature_columns = [
        'Age', 'Duration of Internship (months)', 'Performance Score',
        'Attendance Rate', 'Number of Completed Projects',
        'Technical Skill Rating', 'Soft Skill Rating',
        'Hours Worked per Week', 'Distance from Work (miles)',
        'Recommendation Score', 'Department_Data Science',
        'Department_Finance', 'Department_Human Resources',
        'Department_Marketing', 'Department_Web Development',
        'Socioeconomic Status_High', 'Socioeconomic Status_Low',
        'Socioeconomic Status_Medium', 'Mentorship Level_High',
        'Mentorship Level_Low', 'Mentorship Level_Medium'
    ]

    # Lead conversion model loading
    with open('gb_Fir_model1.sav', 'rb') as model_file:
        lead_model = pickle.load(model_file)

    lead_feature_columns = [
        'Lead Interest Level', 'Course Fee Offered', 'Potential Score',
        'Page Views Per Visit', 'Days Since Last Interaction',
        'Interaction Time-Hour', 'Time Spent on Website', 'Engagement Score',
        'Age'
    ]


def send_notification_email(intern_name, intern_email, prediction_score):
    subject = "Internship Placement Success Notification"
    body = f"""
    Dear {intern_name},

    Congratulations! Based on our recent analysis of your internship performance, we are pleased to inform you that your placement prediction score is {prediction_score}, which indicates a strong likelihood of securing a position with one of our partner organizations.

    We would like to acknowledge your dedication and effort during your internship. As the next step, please expect further communication from us regarding available opportunities and the next steps in the placement process.

    If you have any questions, feel free to reach out. We look forward to seeing you succeed in your future career!

    Best regards,
    Career Development Team
    """

    try:
        msg = Message(
            subject=subject,
            sender=app.config['MAIL_USERNAME'],
            recipients=[intern_email],
            body=body
        )
        mail.send(msg)
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False
def send_notification_email_notplaced(intern_name, intern_email, prediction_score):

    subject = "Internship Placement Status Update"
    body = f"""
    Dear {intern_name},

    Based on our recent analysis of your internship performance, we noticed that there might be some areas for improvement to enhance your placement prospects. Your current placement prediction score is {prediction_score}.

    We recommend scheduling a meeting with your mentor to discuss:
    1. Areas for skill development
    2. Additional project opportunities
    3. Personalized improvement strategies
    Please reply to this email to schedule your consultation.

    Best regards,
    Career Development Team
    """
    try:
        msg = Message(
            subject=subject,
            sender=app.config['MAIL_USERNAME'],
            recipients=[intern_email],
            body=body
        )
        mail.send(msg)
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False

@app.route('/')
def home():
    return render_template('home.html')
def send_email(lead_name, lead_email, prediction):
    if prediction == 1:
        subject = "Welcome to the Course"
        body = f"""
              Dear {lead_name},\n\n

         We are excited to have you onboard! Please find the course details attached and feel free to reach out with any questions.\n\n
        "Best regards,\nAdmissions Team"
         """
    else:
        subject = "Recommendations for Improvement"
        body = f"""
              Dear {lead_name},\n\n

         It seems you haven't converted yet. We recommend attending the following workshops and webinars to enhance your skills:\n"
                        "- Workshop on effective communication\n"
                     "- Webinar on career guidance\n\n"
                     "We look forward to seeing you excel soon!\n\nBest regards,\nAdmissions Team"
         """


    try:
        msg = Message(
            subject=subject,
            sender=app.config['MAIL_USERNAME'],
            recipients=[lead_email],
            body=body
        )
        mail.send(msg)
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False


@app.route('/intern-placement', methods=['GET', 'POST'])
def intern_placement():
    global intern_model, intern_scaler, intern_feature_columns
    prediction_result = None
    email_status = None

    if request.method == 'POST' and 'predict' in request.form:
        if intern_model is not None:
            # Get intern details
            intern_name = request.form.get('intern_name', '')
            intern_email = request.form.get('intern_email', '')
            chkmail = request.form.get('chkmail')
            # Get prediction features
            user_input = {feature: float(request.form.get(feature, 0))
                          for feature in intern_feature_columns}
            input_df = pd.DataFrame([user_input])
            input_scaled = intern_scaler.transform(input_df)

            # Make prediction
            prediction = intern_model.predict(input_scaled)[0]
            prediction_result = round(prediction, 2)

            # Send email if prediction is below threshold
        if chkmail:
            if prediction_result > 0.7 and intern_email:  # Assuming 70 is the threshold
                    email_sent = send_notification_email(
                    intern_name,
                    intern_email,
                    prediction_result
                )
            else:  # Assuming 70 is the threshold
                email_sent = send_notification_email_notplaced(
                    intern_name,
                    intern_email,
                    prediction_result
                )
                # email_status = "Email notification sent successfully" if email_sent else "Failed to send email notification"
            if email_sent:
                email_status="Email notification sent successfully"
            else:
                email_status = "Failed to send email notification"
    return render_template('index.html',
                           prediction_result=prediction_result,
                           email_status=email_status,
                           feature_columns=intern_feature_columns)


@app.route('/lead-conversion', methods=['GET', 'POST'])
def lead_conversion():
    global lead_model, lead_scaler, lead_feature_columns
    prediction_result = None
    email_status1 = None

    if request.method == 'POST' and 'predict' in request.form:
        if lead_model is not None:
            lead_name = request.form.get('lead_name', '')
            lead_email = request.form.get('lead_email', '')
            # email_veri = request.form.get('email_veri', '')
            user_input = {feature: float(request.form.get(feature, 0))
                          for feature in lead_feature_columns}
            input_df = pd.DataFrame([user_input])

            if lead_scaler:
                input_scaled = lead_scaler.transform(input_df)
            else:
                input_scaled = input_df

            prediction = lead_model.predict(input_scaled)[0]
            prediction_result = "Converted" if prediction == 1 else "Not Converted"


            if request.method == 'POST' and 'predict' in request.form:
                chkmail = request.form.get('chkmail')
                if chkmail:


                    try:
                    # Debug print to confirm email details before sending
                        print(f"Sending email to {lead_name} at {lead_email} with prediction: {prediction}")

                    # Attempt to send the email
                        email_sent = send_email(
                        lead_name=lead_name,
                        lead_email=lead_email,
                        prediction=prediction
                        )
                        email_status1 = "Email notification sent successfully" if email_sent else "Failed to send email notification"
                    except TypeError as e:
                        print("Error in send_email:", e)
                        email_status1 = "Error: Failed to send email due to a missing parameter."

    return render_template(
        'lead_conversion.html',
        prediction_result=prediction_result,
        email_status1=email_status1,
        feature_columns=lead_feature_columns
    )


if __name__ == '__main__':
    load_models_and_scalers()
    app.run(debug=True)


