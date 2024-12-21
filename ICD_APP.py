import streamlit as st
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string


# Ensure the app looks in the local nltk_data folder
nltk.data.path.append('./nltk_data')

# Download necessary NLTK resources (only if not already downloaded)
nltk.download('punkt', download_dir='./nltk_data')
nltk.download('stopwords', download_dir='./nltk_data')
nltk.download('wordnet', download_dir='./nltk_data')
nltk.download('punkt_tab', download_dir='./nltk_data')
nltk.download('omw-1.4', download_dir='./nltk_data')  # To fix lemmatizer issues

# Load the dataset
file_path = r'https://raw.githubusercontent.com/jk-vishwanath/Cotiviti_POC/refs/heads/main/Code_with_ouput_and_dataset/icd.csv'  # Replace with your correct file path
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Text preprocessing function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to the 'description' column
data['processed_description'] = data['description'].apply(preprocess_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['processed_description'])

# Train KNN model
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(X)

# Function to get the closest N ICD codes
def get_closest_icd_codes(input_text, N):
    input_text_processed = preprocess_text(input_text)
    input_vector = vectorizer.transform([input_text_processed])
    distances, indices = knn.kneighbors(input_vector, n_neighbors=N)
    closest_ids = data.iloc[indices[0]]['id'].values
    return closest_ids

# Streamlit app
st.title("ICD Code Predictor")
st.write("Enter symptoms to predict the closest ICD codes.")

# Input fields
input_text = st.text_area("Enter symptoms here:")
num_codes = st.number_input("Number of ICD codes to retrieve:", min_value=1, max_value=10, step=1, value=5)

if st.button("Predict"):
    if input_text.strip():
        try:
            closest_codes = get_closest_icd_codes(input_text, num_codes)
            st.write("### Predicted ICD Codes:")
            for code in closest_codes:
                st.write(f"- {code}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter symptoms to predict.")
