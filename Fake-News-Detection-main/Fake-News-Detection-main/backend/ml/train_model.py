import os
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Ensure necessary NLTK data is downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'train.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'ml', 'fake_news_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'ml', 'vectorizer.pkl')

def create_sample_dataset():
    """Creates a sample dataset if train.csv is not found."""
    print("Kaggle train.csv not found. Generating a sample dataset for demonstration...")
    data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8],
        'title': [
            'Aliens discovered on Mars', 
            'Government passes new budget bill',
            'Miracle cure for all diseases found in kitchen',
            'Stock market reaches all time high today',
            'Local man flies by flapping arms',
            'New smartphone released with incredible battery life',
            'Earth is actually flat, say new scientists',
            'Scientists discover new species of deep sea fish'
        ],
        'text': [
            'NASA scientists have officially confirmed that green aliens have been discovered living under the surface of Mars. They have flying saucers.',
            'The senate has passed the new infrastructure and budget bill by a narrow margin. The president is expected to sign it tomorrow.',
            'Doctors hate him! This simple kitchen ingredient cures everything overnight. Just mix salt and vinegar and drink it.',
            'In a historic day for the economy, the stock market indices reached unprecedented highs as tech companies report massive profits.',
            'In an astonishing event, a local man from Florida was seen flying over the city simply by flapping his arms very fast.',
            'The tech giant has unveiled its latest smartphone model featuring a battery that lasts a whole week on a single charge.',
            'A new group of researchers claim that all previous satellite images were faked and the earth is indeed a flat disc.',
            'Marine biologists exploring the Mariana Trench have discovered a new bioluminescent species of fish.'
        ],
        'label': [0, 1, 0, 1, 0, 1, 0, 1] # 0 = Fake, 1 = Real
    }
    df = pd.DataFrame(data)
    # Save the sample so it can be loaded
    df.to_csv(DATA_PATH, index=False)
    return df

def text_cleaning(text):
    """Preprocesses the input text."""
    if not isinstance(text, str):
        return ""
        
    text = text.lower() # Convert to lowercase
    text = re.sub('\[.*?\]', '', text) # Remove text in square brackets
    text = re.sub("\\W"," ",text) # Remove special characters
    text = re.sub('https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub('<.*?>+', '', text) # Remove HTML tags
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # Remove punctuation
    text = re.sub('\n', '', text) # Remove newlines
    text = re.sub('\w*\d\w*', '', text) # Remove words containing numbers
    
    # Stemming and stopword removal
    port_stem = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = text.split()
    text = [port_stem.stem(word) for word in text if not word in stop_words]
    text = ' '.join(text)
    
    return text

def train_model():
    print("Starting model training pipeline...")
    
    # 1. Load Data
    if os.path.exists(DATA_PATH):
        print(f"Loading data from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)
    else:
        df = create_sample_dataset()
        
    # Check if necessary columns exist
    if 'text' not in df.columns or 'label' not in df.columns:
        print("Error: Dataset must contain 'text' and 'label' columns.")
        return

    # Drop NaNs
    df = df.dropna(subset=['text', 'label'])

    print(f"Dataset shape: {df.shape}")
    
    # 2. Text Preprocessing
    print("Preprocessing text data. This may take a while for large datasets...")
    # To speed up demonstration on huge datasets, we could sample, but we'll run on all
    X = df['text'].apply(text_cleaning)
    Y = df['label']
    
    # 3. Feature Extraction (TF-IDF)
    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer()
    vectorizer.fit(X)
    X_vectorized = vectorizer.transform(X)
    
    # 4. Train-Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X_vectorized, Y, test_size=0.2, random_state=42)
    
    # 5. Model Training
    print("Training Multinomial Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train, Y_train)
    
    # 6. Evaluation
    print("Evaluating model...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(Y_test, predictions))
    
    # 7. Save Model
    print("Saving model and vectorizer...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print(f"Model successfully saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
