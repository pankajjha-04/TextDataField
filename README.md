How to get keywords from a text data field in python And how to match the keywords from one dataset with the text field of another dataset

To extract keywords from a text data field in Python, you can use natural language processing libraries such as NLTK (Natural Language Toolkit) or spaCy. Once you have extracted keywords from one dataset, you can match them with the text field of another dataset using various methods like string matching or cosine similarity. Here's a step-by-step guide on how to achieve this:

1. Install and Import Required Libraries:
pip install nltk spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

2. Extract Keywords from Text Data:
You can extract keywords using different techniques. Here are two examples using NLTK and spaCy:

Using NLTK:
def extract_keywords_nltk(text, num_keywords=10):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum()]
    words = [word for word in words if word not in stopwords.words('english')]
    fdist = FreqDist(words)
    keywords = [word for word, freq in fdist.most_common(num_keywords)]
    return keywords
    
Using spaCy:
nlp = spacy.load('en_core_web_sm')

def extract_keywords_spacy(text, num_keywords=10):
    doc = nlp(text)
    keywords = [token.text for token in doc if not token.is_stop and token.is_alpha]
    keywords = [word.lower() for word in keywords]
    fdist = FreqDist(keywords)
    keywords = [word for word, freq in fdist.most_common(num_keywords)]
    return keywords
    
3. Match Keywords between Datasets:
You can use techniques like string matching or cosine similarity to match keywords from one dataset with the text field of another dataset.

String Matching:
def match_keywords_string_matching(keywords1, text2):
    matching_keywords = [keyword for keyword in keywords1 if keyword in text2]
    return matching_keywords

Cosine Similarity:

def match_keywords_cosine_similarity(keywords1, texts2):
    vectorizer = CountVectorizer().fit_transform([', '.join(keywords1)] + texts2)
    vectors = vectorizer.toarray()
    similarity = cosine_similarity([vectors[0]], vectors[1:]).flatten()
    matched_indices = [i for i, sim in enumerate(similarity) if sim > threshold]
    matched_texts = [texts2[i] for i in matched_indices]
    return matched_texts
Note: In the cosine similarity example, you need to define a threshold for similarity based on your data and requirements.

4. Putting It All Together:
Here's how you can use these functions:
# Extract keywords from dataset 1
text1 = "This is a sample text containing keywords"
keywords1 = extract_keywords_nltk(text1)

# Texts in dataset 2
texts2 = ["Another sample text with some keywords",
          "Text without any keywords",
          "This text has a few relevant keywords"]

# Match keywords using cosine similarity
threshold = 0.2
matched_texts = match_keywords_cosine_similarity(keywords1, texts2)

print("Keywords from Dataset 1:", keywords1)
print("Matched Texts:", matched_texts)

Remember to preprocess your text data and adjust parameters according to your specific use case and requirements.
