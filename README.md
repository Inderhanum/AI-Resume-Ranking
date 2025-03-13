# AI-Resume-Ranking
AI-powered Resume Screening and Ranking System
<h1>•	NLTK (Natural Language Toolkit) – For text preprocessing
•	Scikit-Learn – For implementing KNN classification
•	TF-IDF Vectorizer – For feature extraction
•	Streamlit – For building the user interface
•	PyPDF2 – For extracting text from PDFs
•	Gensim – For summarization
</h1>

automate resume screening and ranking using Machine Learning (ML) and Natural Language Processing (NLP). The system follows a structured workflow.
Resume Preprocessing – Extract and clean resume text by removing special characters, stop words, and performing lemmatization.
Feature Extraction – Convert text into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency).
Classification & Similarity Calculation – Use K-Nearest Neighbours (KNN) to classify resumes and Cosine Similarity to determine how well they match job descriptions.
Ranking System – Rank resumes based on their similarity scores to streamline candidate selection for recruiters.
The resume’s provided as input would be shortlisted in this procedure to remove any special or garbage characters from the resumes. All unique characters, numerals, and words with only single letters are eliminated during cleaning. After these processes, we had a clean dataset with no unique characters, numerals, or single letter words. NLTK tokenizers are used to break the dataset into tokens. Stop word removal, lemmatization and vectorization are among the preprocessing operations performed on the tokenized dataset.
The data is masked in the following ways:
•	Masking the strings such as \w 
•	Masking the escape letters like \n 
•	Masking all the numbers 
•	By substituting an empty string for all single-letter words 
•	Stop words are removed • Lemmatization is performed
Removing Stop Words: Stop words such as and, the, was, and others appear very often in words and limit the process which determines prediction thus they are removed. Filtering the Stop Words consists of the following steps: 
1.	The input words are tokenized into individual tokens and saved in an array. 
2.	Each word now corresponds to the Stop Words list in the NLTK library.
(a)	import stopwords from nltk. corpus
(b)	SW[] = set(stopwords.words('english'))
(c)	It returns 180 stop words, which may be confirmed using the (len(StopWords)) function. and displayed using the print (StopWords) function.
3.	When the words appear in the StopWords list, they are removed from the main sentence array. 
4.	Repeat the above sequence of steps until the tokenized array's last entry is not matched. 
5.	There are no stop words in the resultant array.

The extraction of features is the next phase. We used the Tf Idf (Term Frequency, Inverse Document Frequency) to extract features from a preprocessed dataset. The cleansed data was transferred, and Tf-Idf was used to extract features. Taking the input as a numerical vector, processing a machine learning based classification model or learning algorithms takes place. The input text with varying length was not processed by ML based classifiers. As a result, during the preparation procedures, the texts are changed to the required equal length vector form. There are several methods for extracting characteristics, including tf-idf, and others. Using the scikit learn library function, we generated tf-idf for each term. To calculate a tf-idf vector,we use TfidfVectorizer: 
1.	Sub-linear df is set to True to utilise a logarithmic form for frequency. 
2.	Min df is the minimal number of documents in which a word must appear in order to be saved. 
3.	The norm is set to l2 to ensure that all feature vectors have the same euclidean norm.
4.	The gramme range is set to (n1, n2), with n1 equaling 1 and n2 equaling 2. It means that both unigrams and bigrams are taken into account.
The model takes a job description and a list of CVs as input and returns a list of CVs that are most similar to the job description. 
Given that this is a situation of document similarity detection, I've chosen the Cosine Similarity Algorithm, in which the employer's Job Description is matched against the content of resumes in the space, and the top most similar resumes are suggested to the recruiter. The algorithm merges the cleaned resume data and job description into an unified set of data before computing the cosine similarity between both the job description and CVs. 
The k-NN model is used in this model to find the resumes that are closest to the specified job description. To begin, we utilised an open source tool called "gensim" to scale the JD and CVs. The package used gives a summary of the given text within the limit of words that is provided. To get the JD and resumes to the same word scale, this library was used to build a summary of the JD and CVs, and then k-NN was used to locate CVs that closely matched the given JD.
Sumy is one of the Python libraries for Natural Language Processing tasks. It is mainly used for automatic summarization of paragraphs using different algorithms.
•	Sumy provides many summarization algorithms, allowing users to choose from a wide range of summarizers based on their preferences.
•	This library integrates efficiently with other NLP libraries.
•	The library is easy to install and use, requiring minimal setup.
•	We can summarize lengthy documents using this library.
•	Sumy can be easily customized to fit specific summarization needs.

<h3>How to run</h3>
After downloading the raw code save the code in a particular folder
open your command "cmd" the cmd will be in the user.
first copy your folder path
change it to the folder path by using change directore "cd" and paste the folder path
HERE COMES THE MAIN THING HOW TO RUN THE FILE
**PLEASE MAKE SURE YOUR FILE NAME DOESNOT HAVE ANY SPACING**
****FOLLOW THIS ***
<b>streamlit run file_name.py</b>
****IN MY CODE I HAD SAVED WITH ai_resume_app SO I WILL RUN IN THIS WAY****
<b>streamlit run ai_resume_app.py</b>
![Screenshot 2025-03-06 231105](https://github.com/user-attachments/assets/7cd9c761-6426-4b4d-bc8a-4cd6cbdfff6d)

![Screenshot 2025-03-06 231306](https://github.com/user-attachments/assets/e0fbae58-7e23-4952-bbf5-a44b0c11e235)
