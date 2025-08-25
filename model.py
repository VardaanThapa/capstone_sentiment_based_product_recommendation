import pandas as pd
import pickle as pk
import re, string
import nltk
from nltk.stem import WordNetLemmatizer
from paths import DATASETS_DIR, OUTPUTS_DIR, RESULTS_DIR

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

class Model:
  def __init__(self):
    self.dataset = pd.read_csv(DATASETS_DIR + "/sample30.csv")
    # Assign meaningful column names
    self.dataset = self.dataset.rename(columns={"name" : "product"})
    # Selecting only relevant columns and Removing all the unnecessary columns
    self.dataset = self.dataset[[ "reviews_username", "product", "reviews_rating", "reviews_text", "user_sentiment"]]

    self.lemmatizer = WordNetLemmatizer()
    self.stop_words = set(stopwords.words('english'))
    
    self.tfidf_vectorizer = pk.load(open(OUTPUTS_DIR + "/tfidf_vectorizer.pkl", "rb"))

    self.model = pk.load(open(RESULTS_DIR + "/final_xgb_model.pkl", "rb"))
    self.recommendation = pk.load(open(OUTPUTS_DIR + "/recommendation.pkl", "rb"))


 # function to perform text processing (Lemmatize and Remove stop words)
  def __process_text(self, text):
      """Cleans and processes text data."""
      text = re.sub('[^a-zA-Z]', ' ', text)
      text = text.lower()
      text = text.split()
      # Apply lemmatization and remove stopwords
      text = [self.lemmatizer.lemmatize(word) for word in text if word not in self.stop_words]
      return ' '.join(text)


  def check_username(self, username):
    row_count = len(self.dataset[self.dataset["reviews_username"] == username])
    print("check_username :: row count = {}".format(row_count))
    if row_count == 0:
      return "{} not found in the dataset".format(username)
    else:
      return None


  # Find the top 20 recommended products for a user
  def top_20_recommended_products(self, username):
    if username not in self.recommendation.index:
      return None

    product_list = self.recommendation.loc[username].sort_values(ascending=False)[0:20]
    print("Top 20 recommended_products for {}".format(username))
    display(product_list.index)
    print()
    products = self.dataset[self.dataset["product"].isin(product_list.index.tolist())]
    products = products[['product', 'reviews_text']]
    return products


  # Fine-tune the recommended products and return products with highest positive sentiment
  def top_5_products(self, products):
    products["review_processed"] = products["reviews_text"].apply(self.__process_text)
    
    # Create TF-IDF vectors from the text
    X_tfidf = self.tfidf_vectorizer.transform(products["review_processed"])

    # sentiment prediction
    products['predicted_sentiment'] = self.final_xgb_model.predict(X_tfidf)
    
    total_product = products.groupby(['product']).agg('count')
    recommended_df = products.groupby(['product','predicted_sentiment']).agg('count')
    recommended_df = recommended_df.reset_index()
    merged_df = pd.merge(recommended_df, total_product['reviews_text'], on='product')
    merged_df['percentage'] = (merged_df['reviews_text_x']/merged_df['reviews_text_y'])*100
    merged_df = merged_df.sort_values(ascending=False, by='percentage')
    
    return list(merged_df[merged_df['predicted_sentiment'] == 1]['product'][:5])
  

model = Model()