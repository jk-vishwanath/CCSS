import streamlit as st
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Load pre-trained model and vectorizer
with open("knn_model.pkl", "rb") as knn_file:
    knn = pickle.load(knn_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load dataset for ID retrieval (must match the dataset used to train the model)
data = pd.read_csv("https://raw.githubusercontent.com/jk-vishwanath/CCSS/refs/heads/main/ccss.csv")

# Preprocessing function
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


# Streamlit app
st.title("CCSS Code Alignment Tool")

input_text = st.text_area("Enter Text for Alignment:", "")
n_neighbors = st.number_input("Number of Closest Matches (N):", min_value=1, value=5)

if st.button("Get Closest CCSS IDs"):
    if input_text:
        try:
            # Preprocess input text
            input_text_processed = preprocess_text(input_text)

            # Transform input text using the TF-IDF vectorizer
            input_vector = vectorizer.transform([input_text_processed])

            # Get the closest N neighbors
            distances, indices = knn.kneighbors(input_vector, n_neighbors=n_neighbors)

            # Retrieve the closest CCSS IDs
            closest_ids = data.iloc[indices[0]]["id"].values

            # Display the results
            st.write("Closest CCSS IDs:", closest_ids)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please provide input text!")
