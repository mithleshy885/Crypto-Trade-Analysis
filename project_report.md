# Crypto Trading Sentiment & Pattern Analysis Project Report

## Project Overview
This project analyzes crypto trading data in conjunction with market sentiment (Fear-Greed Index) to uncover actionable insights, segment traders, and build predictive models. The workflow includes data preprocessing, exploratory analysis, sentiment-performance analysis, trader segmentation, machine learning classification, and trade pattern recognition.

## Data Sources
- **historical_data.csv**: Contains trade-level data (PnL, size, fee, account, etc.)
- **fear_greed_index.csv**: Contains daily market sentiment (Extreme Fear, Fear, Neutral, Greed, Extreme Greed)

## Data Preprocessing & Feature Engineering
- Timestamps are parsed and aligned by date.
- Sentiment and trading data are merged on date.
- Features engineered: trade hour, is_profitable, abs_pnl, rolling means/stds, lagged sentiment.
- Data cleaning: missing values handled, outliers can be visualized.

## Sentiment Analysis & Insights
- **Exploratory Analysis**: Visualizes sentiment distribution, PnL by sentiment, trade frequency, win rate, average trade size, and side preference.
- **Performance by Sentiment**: Computes mean, median, std, and sum of PnL, win rate, and trade size for each sentiment class.
- **Key Insights**:
  - Certain sentiments (e.g., Greed/Extreme Greed) may correlate with higher average PnL or win rates.
  - Risk-reward and trade size vary by sentiment.

## Trader Segmentation & Responsiveness
- **Segmentation**: Traders are grouped by total PnL (High Loss, Small Loss, Small Profit, High Profit) and activity level (Low, Medium, High, Very High).
- **Analysis**: Examines how different trader segments respond to sentiment, and their trade frequency.
- **Insights**:
  - High-profit traders may behave differently under Greed vs. Fear.
  - Activity level impacts sentiment responsiveness.

## ML Classification Model
- **Goal**: Predict tomorrow's sentiment using today's trading features and sentiment.
- **Features Used**: Closed PnL, Size USD, Fee, trade hour, is_profitable, abs_pnl, rolling means/stds, lagged sentiment.
- **Model**: RandomForestClassifier with hyperparameter tuning (GridSearchCV).
- **Usage**:
  - Train: Run `app.py` to train and evaluate the model. Model is saved as `sentiment_classifier.pkl`.
  - Predict: Use `gui.py` (Streamlit app) to input features and get a prediction for tomorrow's sentiment.
- **Results**: Classification report and confusion matrix are printed after training.
- **Recommendations**: Focus trading during favorable sentiment periods, adjust trade size and direction by sentiment, and manage risk accordingly.

## Trade Pattern Recognition (KMeans Clustering)
- **Goal**: Discover patterns in trading behavior using clustering.
- **Features Used**: Closed PnL, Size USD, Fee, trade hour, is_profitable, abs_pnl.
- **Method**: KMeans clustering (optimal k found via elbow method).
- **Usage**:
  - Run `trade_pattern_recognition.py` to cluster trades and visualize patterns.
  - Cluster assignments are saved to `trading_patterns_with_clusters.csv`.
- **Insights**:
  - Clusters reveal distinct trading styles (e.g., high PnL/large size vs. low PnL/small size).
  - Cluster centers help interpret typical behaviors.

## Usage Instructions
- **Data Preparation**: Place `historical_data.csv` and `fear_greed_index.csv` in the project folder.
- **Analysis**: Run `trading_sentiment_analysis.py` for full EDA and insights.
- **ML Model**: Run `app.py` to train the classifier. Use `gui.py` for predictions.
- **Pattern Recognition**: Run `trade_pattern_recognition.py` for clustering and pattern discovery.

## Key Insights & Recommendations
- Sentiment strongly influences trading outcomes; Greed periods may offer higher returns but also higher risk.
- Trader segmentation reveals that high-activity, high-profit traders respond differently to sentiment than others.
- Machine learning models can provide actionable predictions for trading strategy.
- Pattern recognition helps identify and monitor distinct trading behaviors for further analysis or risk management.

---
*This report was generated automatically. For further details, see the code and visualizations in the project files.*
