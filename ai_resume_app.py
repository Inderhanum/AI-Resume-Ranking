import streamlit as st
import pandas as pd
import numpy as np
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Download NLTK dependencies
nltk.download('punkt')
nltk.download('stopwords')

# Streamlit app settings
st.set_page_config(page_title="Resume Ranker", layout="wide")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to summarize text using sumy
def summarize_text(text, sentence_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)

# Function to clean and summarize text
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    clean_text = ' '.join(tokens)
    
    # Summarize if text exceeds 100 words
    if len(clean_text.split()) > 100:
        return summarize_text(clean_text)
    return clean_text

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Resume Ranking", "About"])

if page == "Home":
    st.title("Welcome to Resume Ranker")
    st.write("An AI-powered resume ranking system to match candidates with job descriptions effectively. Welcome to Resume Ranker ")
    st.write("🚀An AI-powered Resume Screening and Ranking System")
    st.write("Finding the right candidate for a job can be challenging, especially when dealing with hundreds or even thousands of resumes. Traditional resume screening is time-consuming, prone to bias, and often results in missed opportunities. Resume Ranker is an AI-driven solution designed to streamline and enhance the recruitment process.")

   

    st.write("How It Works?")
    st.write("✅ Upload Job Description – Paste or upload the job details")
    st.write("✅ Upload Resumes – Add multiple resumes in PDF format")
    st.write("✅ AI Processing – The system extracts key information, compares resumes with job requirements, and ranks them")
    st.write("✅ View Ranked Resumes – Get a sorted list based on relevance and similarity score")

    st.write("Why Choose Resume Ranker? ")
    st.write("✅ Saves Time – Automates manual resume screening")
    st.write("✅ Improves Accuracy – Uses Machine Learning & NLP to enhance job-candidate matching")
    st.write("✅ Easy to Use – Simple drag-and-drop interface for uploading resumes")
    st.write("✅ Customizable – Adaptable to various industries and hiring needs")

    st.write("🚀 Start using Resume Ranker today and hire the best candidates faster!")
elif page == "Resume Ranking":
    st.title("Resume Classification & Ranking")
    
    # Upload job description
    st.subheader("Upload Job Description")
    job_description = st.text_area("Paste the Job Description here")
    
    # Upload resumes
    st.subheader("Upload Resumes")
    uploaded_files = st.file_uploader("Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)
    
    if job_description and uploaded_files:
        st.write("Processing resumes...")
        resume_texts = []
        resume_names = []
        
        for uploaded_file in uploaded_files:
            text = extract_text_from_pdf(uploaded_file)
            clean_text = preprocess_text(text)
            resume_texts.append(clean_text)
            resume_names.append(uploaded_file.name)
        
        # Preprocess job description
        job_desc_clean = preprocess_text(job_description)

        # Vectorization
        vectorizer = TfidfVectorizer()
        all_texts = resume_texts + [job_desc_clean]
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # Train KNN Classifier
        labels = np.arange(len(resume_texts))  # Assign labels (dummy classification)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(tfidf_matrix[:-1], labels)

        # Predict categories
        predictions = knn.predict(tfidf_matrix[:-1])

        # Compute Cosine Similarity
        similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

        # Evaluate Model
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='weighted')
        recall = recall_score(labels, predictions, average='weighted')
        f1 = f1_score(labels, predictions, average='weighted')
        kappa = cohen_kappa_score(labels, predictions)

        # Display Evaluation Metrics
        st.subheader("Model Evaluation Metrics")
        st.metric("Accuracy", f"{accuracy:.2f}")
        st.metric("Precision", f"{precision:.2f}")
        st.metric("Recall", f"{recall:.2f}")
        st.metric("F1-score", f"{f1:.2f}")
        st.metric("Kappa Score", f"{kappa:.2f}")

        # Create results DataFrame
        results_df = pd.DataFrame({
            "Resume": resume_names,
            "Predicted Category": predictions,
            "Similarity Score": similarity_scores
        }).sort_values(by="Similarity Score", ascending=False)

        # Display results
        st.subheader("Ranked Resumes")
        st.dataframe(results_df.style.format({"Similarity Score": "{:.2f}"}).background_gradient(cmap="Blues"))

elif page == "About":
    st.title("About Resume Ranker")
    
    st.write("🚀 **AI-Powered Resume Screening and Ranking Tool**")
    
    st.write("""
    Resume Ranker is an AI-powered system designed to simplify and enhance the recruitment process. 
    It leverages **Machine Learning (ML)** and **Natural Language Processing (NLP)** techniques 
    like **TF-IDF**, **k-Nearest Neighbors (k-NN)**, and **Cosine Similarity** to analyze and rank 
    resumes based on their relevance to job descriptions.
    
    **Key Features:**
    - Automated resume parsing and keyword extraction
    - Relevance scoring using AI models
    - Candidate ranking for quick decision-making
    - Performance evaluation metrics (Accuracy, Precision, Recall, etc.)
    
    **How It Works:**
    1. Upload Job Description
    2. Upload multiple resumes in PDF format
    3. The system processes and ranks resumes based on their similarity to the job description
    
    ---
    
    ### About the Developer 👨‍💻
    
    **Inder Hanumanthkari**  
    📍 MCA Graduate from Jawaharlal Nehru Technological University, Anantapur (CGPA: 7.09)  
    📞 +91-9133216623  
    🔗 [LinkedIn](https://www.linkedin.com/in/inder-hanumanthkari-a050bb293/)  
    💻 [GitHub](https://github.com/Inderhanum)
    
    I am a motivated software developer with a solid foundation in **Python**, **HTML**, **CSS**, 
    **JavaScript**, **React.js**, and **SQL**. I am passionate about developing dynamic web applications 
    and intuitive user interfaces, with a keen interest in learning new technologies.
    
    **Technical Skills:**
    - Languages: Python, C, C++, JavaScript, HTML, CSS
    - Frameworks & Tools: React.js, Bootstrap, Git, GitHub
    
    **Soft Skills:**
    - Communication
    - Teamwork
    - Problem-solving
    - Time Management
    
    ---
    
    ### Projects:
    
    ✅ **AI-Powered Resume Ranking System (Streamlit, Python, NLP):**  
    - Developed an AI-driven system that ranks resumes based on job descriptions  
    - Utilized TF-IDF, k-NN, and Cosine Similarity for classification and ranking  
    - Implemented with Streamlit for an interactive user interface
    
    ✅ **To-Do List (React.js):**  
    - Created a dynamic to-do list app with task management features (Add/Edit/Delete tasks)  
    
    ✅ **Mobile Phone Selection App (Python):**  
    - Command-line Python app guiding users through mobile phone selection based on preferences  
    
    ✅ **D-Mart Web Application (HTML, CSS, Bootstrap):**  
    - Developed a static e-commerce webpage with product listings and navigation  
    
    ✅ **Rock Paper Scissors Game (Python):**  
    - Built an interactive command-line game using Python’s random module  
    
    ---
    
    ✨ Eager to contribute to collaborative environments and grow within the tech industry!
    """)
