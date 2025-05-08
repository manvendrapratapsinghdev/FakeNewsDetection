import streamlit as st
import joblib
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class FakeNewsDetectionApp:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.abspath(os.path.join(self.script_dir, "../model/model/logistic_regression_model.pkl"))
        self.vectorizer_path = os.path.abspath(os.path.join(self.script_dir, "../model/model/tfidf_vectorizer.pkl"))
        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)

        # Initialize session state for 'show_info'
        if "show_info" not in st.session_state:
            st.session_state["show_info"] = False

    def render_title(self):
        st.title("Vishleshak: Fake News Detection APP")

        st.write("Enter a news article below to check if it's fake or real.")
    def get_user_input(self):
        # Bind input fields to session state variables
        if "Title" not in st.session_state:
            st.session_state["Title"] = ""
        if "Text" not in st.session_state:
            st.session_state["Text"] = ""
        if "Category" not in st.session_state:
            st.session_state["Category"] = ""
        if "Date" not in st.session_state:
            st.session_state["Date"] = None

        title = st.text_input("Title", value=st.session_state["Title"])
        text = st.text_area("Text", value=st.session_state["Text"])
        category = st.text_input("Category (Sport, Politics, etc.)", value=st.session_state["Category"])
        date = st.date_input("Date", value=st.session_state["Date"])

        # Update session state with current input values
        st.session_state["Title"] = title
        st.session_state["Text"] = text
        st.session_state["Category"] = category
        st.session_state["Date"] = date

        return title, text, category, date

    def predict_news(self, user_input):
        input_features = self.vectorizer.transform([user_input])
        return self.model.predict(input_features)[0]

    def display_result(self, prediction):
        if prediction == 1:
            st.success("The news article is Real.")
        else:
            st.error("The news article is Fake.")

    def log_article(self, title, text, subject, date, prediction):
        log_file = os.path.join(self.script_dir, "article_log.csv")
        log_data = {
            "title": [title],
            "text": [text],
            "subject": [subject if subject.strip() else "Unknown"],
            "date": [date],
            "label": ["Real" if prediction == 1 else "Fake"],
            "text_length": [len(text)]
        }
        log_df = pd.DataFrame(log_data)

        if os.path.exists(log_file):
            log_df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            log_df.to_csv(log_file, index=False)

    def log_feedback(self, title, text, prediction, feedback):
        feedback_file = os.path.join(self.script_dir, "feedback_log.csv")
        feedback_data = {
            "title": [title],
            "text": [text],
            "prediction": ["Real" if prediction == 1 else "Fake"],
            "feedback": [feedback]
        }
        feedback_df = pd.DataFrame(feedback_data)

        if os.path.exists(feedback_file):
            feedback_df.to_csv(feedback_file, mode='a', header=False, index=False)
        else:
            feedback_df.to_csv(feedback_file, index=False)

    def run(self):
        self.render_title()
        title, text, subject, date = self.get_user_input()

        col1, _ = st.columns(2)

        with col1:
            if st.button("Check News"):
                if title.strip() == "" or text.strip() == "":
                    st.warning("Please enter both title and text to analyze.")
                else:
                    prediction = self.predict_news(title + text)
                    self.display_result(prediction)
                    self.log_article(title, text, subject, date, prediction)

                    # Store prediction in session state for feedback
                    st.session_state.last_prediction = {
                        "title": title,
                        "text": text,
                        "prediction": prediction
                    }

# Moved the 'About the App' and 'Instructions' sections to the sidebar
        st.sidebar.markdown("## About the App")
        st.sidebar.warning("Note: The model is trained using data from sources like Reuters (https://www.reuters.com/). It may exhibit bias towards this dataset. For in-house use, consider retraining the model with your own data.")

        st.sidebar.markdown("## Instructions")
        st.sidebar.markdown("1. Enter the title and text of the news article.")
        st.sidebar.markdown("2. Select the category (optional).")
        st.sidebar.markdown("3. Click the 'Check News' button to analyze the article.")
        st.sidebar.markdown("4. Provide feedback on the prediction (optional).")
        st.sidebar.markdown("5. The result will be displayed below the button.")

# Run the app
if __name__ == "__main__":
    app = FakeNewsDetectionApp()
    app.run()
