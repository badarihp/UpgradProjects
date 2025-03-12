# Importing Libraries
import pandas as pd
import re, nltk, spacy
import pickle as pk
import time
import logging
import numpy as np
import time

from sklearn.model_selection import RandomizedSearchCV

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')

# Importing stopwords and stemming/lemmatization tools from nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Downloading necessary nltk datasets for text processing
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Download the model
try:
    spacy.cli.download("en_core_web_sm")
    logging.info("Model downloaded successfully.")
except Exception as e:
    logging.info(f"Error downloading model: {e}")

# Loading pre-trained models and vectorizers from pickle files
tfidf_vectorizer = pk.load(open('pickle_file/tfidf_vectorizer.pkl', 'rb'))  # Pre-trained TF-IDF Transformer
rf_model = pk.load(open('pickle_file/rf_base_model.pkl', 'rb'))  # Random Forest Model
lgbm_base_model = pk.load(open('pickle_file/lgbm_base_model.pkl', 'rb'), encoding='latin1')  # XGBoost HPT RS Model


recommend_matrix = pk.load(open('pickle_file/user_final_rating.pkl', 'rb'))  # Pre-trained user-user recommendation matrix

# Loading the spaCy language model with unnecessary components disabled for efficiency
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

# Reading sample product data
product_df = pd.read_csv('sample30.csv', sep=",")

# Global Caches for Recommendations
top5_cache = {}  # Dictionary to store cached top 5 products

# Verify if product_df contains the necessary columns

# Function to preprocess text
def normalize_and_lemmatize(input_text):
    input_text = re.sub(r'[^a-zA-Z\s]', '', input_text)
    words = nltk.word_tokenize(input_text.lower())
    words = [word for word in words if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word, pos='v') for word in words])

# Function to predict sentiment
def model_predict(text_list, model_type):
    if not text_list:
        logging.warning("No text data available for sentiment prediction.")
        return []
    logging.info(f"Predicting sentiment for {len(text_list)} reviews...")
    tfidf_vectors = tfidf_vectorizer.transform(text_list)
    model = rf_model if model_type == 'rf_base' else lgbm_base_model
    predictions = model.predict(tfidf_vectors)
    # logging.info(f"Sentiment predictions: {predictions}")
    return predictions.tolist()

def cached_recommendations(user_name, model_type):
    cache_key = f"{user_name}_{model_type}"
    if cache_key in top5_cache:
        return top5_cache[cache_key], True  # Returning cached result with flag

    logging.info(f"Fetching recommendations for user: {user_name}")
    if user_name not in recommend_matrix.index:
        logging.warning(f"User {user_name} does not exist in recommendation matrix.")
        return []
    
    # Increase recommendations to 50 before filtering
    sorted_recommendations = recommend_matrix.loc[user_name].sort_values(ascending=False)
    
    top_recommended_products = list(sorted_recommendations[:20].index)
    # logging.info(f"Top 20 recommended products: {top_recommended_products}")
    
    # Ensure correct filtering column exists
    filter_col = 'id' if 'id' in product_df.columns else 'name'
    available_products = set(product_df[filter_col].unique())
    missing_products = [prod for prod in top_recommended_products if prod not in available_products]
    
    
    df_top_products = product_df[product_df[filter_col].isin(top_recommended_products)]
    logging.info(f"Products remaining after filtering: {len(df_top_products)}")
    
    # Ensure diversity in selection by taking at most 10 reviews per product
    df_top_products = df_top_products.groupby('name').head(10).reset_index(drop=True)
    logging.info(f"Unique products before aggregation: {df_top_products['name'].nunique()}")
    
    if df_top_products.empty:
        logging.warning("No valid reviews found for recommended products.")
        return []
    
    logging.info(f"Processing {len(df_top_products)} products with reviews...")
    df_top_products['reviews_lemmatized'] = df_top_products['reviews_text'].map(normalize_and_lemmatize)
    reviews_list = df_top_products['reviews_lemmatized'].tolist()
    
    
    if not any(reviews_list):
        logging.warning("All reviews were removed during preprocessing.")
        return []
    
    df_top_products['predicted_sentiment'] = model_predict(reviews_list, model_type)
    
    
    # Aggregate using mode instead of concatenation
    pred_df = df_top_products.groupby(by='name').agg(
        pos_sent_count=('predicted_sentiment', lambda x: sum(1 for i in x if i == 'Positive')),
        total_sent_count=('predicted_sentiment', 'count'),
        post_sent_percentage=('predicted_sentiment', lambda x: np.round(sum(1 for i in x if i == 'Positive') / len(x) * 100, 2)),
        most_common_sentiment=('predicted_sentiment', lambda x: x.mode()[0] if not x.mode().empty else 'Neutral')
    )
    
    # Ensure product names are retained
    pred_df.reset_index(inplace=True)
    
    # Convert DataFrame to nested dictionary with only required columns
    recommendations = pred_df.sort_values(by=['post_sent_percentage', 'total_sent_count'], ascending=[False, False]).head(5).to_dict(orient='records')
    logging.info(f"Final Recommendations Sent to Flask for user {user_name}: {recommendations}")
    
    # Store in cache
    top5_cache[cache_key] = recommendations
    
    return recommendations, False


# Updated recommendation function
def product_recommendations_user(user_name, model_type):
    start_time = time.time()
    recommendations, from_cache = cached_recommendations(user_name, model_type)
    end_time = time.time()
    latency = end_time - start_time
    print(f"Time taken {'(from cache)' if from_cache else '(computed)'} using model: {model_type}, for user {user_name}: {latency:.4f} seconds")
    return recommendations, from_cache