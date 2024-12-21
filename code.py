import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Title of the app
st.title("TF-IDF Calculator")

# Upload the document
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file:
    # Read the file
    document = uploaded_file.read().decode("utf-8")
    
    # Display the content of the file
    st.subheader("Uploaded Document")
    st.write(document)

    # Split the document into lines (if needed)
    lines = document.split("\n")
    
    # Initialize TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Compute TF-IDF
    tfidf_matrix = vectorizer.fit_transform(lines)
    
    # Extract feature names (words) and convert TF-IDF matrix to a dense array
    feature_names = vectorizer.get_feature_names_out()
    dense_matrix = tfidf_matrix.toarray()
    
    # Create a DataFrame to display the results
    tfidf_df = pd.DataFrame(dense_matrix, columns=feature_names, index=[f"Line {i+1}" for i in range(len(lines))])

    # Display the TF-IDF scores as a table
    st.subheader("TF-IDF Scores")
    st.write(tfidf_df)

    # Allow downloading the table as a CSV file
    csv = tfidf_df.to_csv(index=True)
    st.download_button(
        label="Download TF-IDF Table as CSV",
        data=csv,
        file_name="tfidf_scores.csv",
        mime="text/csv",
    )
