Code Explanation
1. Importing Libraries
We begin by importing necessary libraries for data manipulation, analysis, visualization, and machine learning:
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


2. Load the Data
We load the transaction and feedback data from CSV files:
# Load the data
transactions = pd.read_csv('transactions.csv')
feedback = pd.read_csv('customer_feedback.csv')


3. Data Cleaning
We clean the data to handle missing values and convert date columns to the appropriate format:
# Data Cleaning
transactions.dropna(inplace=True)
feedback.dropna(inplace=True)
transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
feedback['date'] = pd.to_datetime(feedback['date'])

4. Customer Segmentation
We segment customers based on their transaction behavior:
# Customer Segmentation
customer_data = transactions.groupby('customer_id').agg({
    'transaction_amount': ['sum', 'mean'],
    'transaction_date': 'count'
}).reset_index()
customer_data.columns = ['customer_id', 'total_spent', 'average_spent', 'transaction_count']

kmeans = KMeans(n_clusters=4, random_state=0)
customer_data['segment'] = kmeans.fit_predict(customer_data[['total_spent', 'average_spent', 'transaction_count']])


5. Behavioral Analysis
We visualize customer spending behavior by segment:
# Behavioral Analysis
plt.figure(figsize=(12, 6))
for segment in customer_data['segment'].unique():
    segment_data = customer_data[customer_data['segment'] == segment]
    plt.plot(segment_data.index, segment_data['total_spent'], label=f'Segment {segment}')

plt.xlabel('Index')
plt.ylabel('Total Spent')
plt.title('Customer Spending by Segment')
plt.legend()
plt.show()


6. Personalized Recommendations
We develop a recommendation system based on customer transaction history:
# Personalized Recommendations
product_data = transactions.pivot_table(index='customer_id', columns='product_id', values='transaction_amount', fill_value=0)
product_similarity = cosine_similarity(product_data)

def recommend_products(customer_id, product_data, product_similarity, n_recommendations=5):
    customer_idx = product_data.index.get_loc(customer_id)
    similarity_scores = product_similarity[customer_idx]
    product_indices = similarity_scores.argsort()[-n_recommendations:][::-1]
    recommended_products = product_data.columns[product_indices]
    return recommended_products

customer_id = 1  # Change this to test with different customers
recommended_products = recommend_products(customer_id, product_data, product_similarity)
print(f'Recommended products for customer {customer_id}: {recommended_products}')


7. Sentiment Analysis
We analyze customer feedback to determine the sentiment:
# Sentiment Analysis
def analyze_sentiment(feedback_text):
    analysis = TextBlob(feedback_text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

feedback['sentiment'] = feedback['feedback'].apply(analyze_sentiment)
print(feedback[['feedback', 'sentiment']].head())
8. Targeted Marketing Campaigns
We create and evaluate a machine learning model to predict customer response to marketing campaigns:
# Create dummy columns for the new columns required for sentiment analysis, here assuming some values for transaction_count, total_spent and average_spent
feedback['transaction_count'] = feedback['customer_id'].map(customer_data.set_index('customer_id')['transaction_count'])
feedback['total_spent'] = feedback['customer_id'].map(customer_data.set_index('customer_id')['total_spent'])
feedback['average_spent'] = feedback['customer_id'].map(customer_data.set_index('customer_id')['average_spent'])

X = feedback[['transaction_count', 'total_spent', 'average_spent']]
y = feedback['campaign_response']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

new_campaign_data = pd.DataFrame({
    'transaction_count': [5, 15, 30],
    'total_spent': [500, 2000, 5000],
    'average_spent': [100, 133.33, 166.67]
})

new_campaign_predictions = model.predict(new_campaign_data)
print('Campaign Response Predictions:', new_campaign_predictions)




This README.md file provides a comprehensive guide to understanding and using the provided code for customer insights and personalization in the banking sector.








