import streamlit as st
import requests

st.title("CCSS Code Alignment Tool")

input_text = st.text_area("Enter Text for Alignment:", "")
n_neighbors = st.number_input("Number of Closest Matches (N):", min_value=1, value=5)

if st.button("Get Closest CCSS IDs"):
    if input_text:
        url = "http://127.0.0.1:5000/get_closest_ids"  # Replace with deployed API URL if hosted elsewhere
        payload = {"text": input_text, "N": n_neighbors}
        
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                st.write("Closest CCSS IDs:", response.json().get("closest_ids"))
            else:
                st.error(f"Error: {response.json().get('error')}")
        except Exception as e:
            st.error(f"Failed to connect to the API. Error: {str(e)}")
    else:
        st.warning("Please provide input text!")
