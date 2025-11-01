import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load merged data from TradingSentimentAnalyzer
from trading_sentiment_analysis import TradingSentimentAnalyzer

# Initialize and prepare data
analyzer = TradingSentimentAnalyzer()
analyzer.load_data('historical_data.csv', 'fear_greed_index.csv')
analyzer.preprocess_data()
analyzer.merge_datasets()

# Prepare features and target for classification
# Example: Predict next day's sentiment based on today's trading features
merged = analyzer.merged_df.copy()
merged = merged.sort_values('datetime')

# Shift sentiment to next day for prediction
date_sentiment = merged[['date', 'classification']].drop_duplicates().sort_values('date')
date_sentiment['next_sentiment'] = date_sentiment['classification'].shift(-1)
merged = merged.merge(date_sentiment[['date', 'next_sentiment']], on='date', how='left')

# Drop rows where next_sentiment is NaN
merged = merged.dropna(subset=['next_sentiment'])

# Feature engineering: add rolling mean/volatility features for PnL and trade size
merged['pnl_rolling_mean_3'] = merged['Closed PnL'].rolling(window=3, min_periods=1).mean()
merged['pnl_rolling_std_3'] = merged['Closed PnL'].rolling(window=3, min_periods=1).std().fillna(0)
merged['size_rolling_mean_3'] = merged['Size USD'].rolling(window=3, min_periods=1).mean()
merged['size_rolling_std_3'] = merged['Size USD'].rolling(window=3, min_periods=1).std().fillna(0)

# Add lagged sentiment as a feature (encode as ordinal)
sentiment_map = {s: i for i, s in enumerate(analyzer.SENTIMENT_ORDER)}
merged['sentiment_lag1'] = merged['classification'].map(sentiment_map)

# Updated feature list
features = [
    'Closed PnL', 'Size USD', 'Fee', 'trade_hour', 'is_profitable', 'abs_pnl',
    'pnl_rolling_mean_3', 'pnl_rolling_std_3', 'size_rolling_mean_3', 'size_rolling_std_3',
    'sentiment_lag1'
]
X = merged[features]
y = merged['next_sentiment']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameter tuning with GridSearchCV 
param_grid = {
    'n_estimators': [100],
    'max_depth': [None, 10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': ['sqrt']
}
gs = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),
                  param_grid, cv=2, n_jobs=-1, scoring='f1_weighted', verbose=1)
gs.fit(X_train, y_train)
clf = gs.best_estimator_

# Evaluate
y_pred = clf.predict(X_test)
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Save model
# joblib.dump(clf, 'sentiment_classifier.pkl')
# print('Model saved as sentiment_classifier.pkl')
