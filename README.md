# Crypto Trading Sentiment & Pattern Analysis

## Overview
This project analyzes crypto trading data and market sentiment (Fear-Greed Index) to uncover actionable insights, segment traders, build predictive models, and recognize trade patterns. It includes:
- Data preprocessing and feature engineering
- Exploratory and sentiment analysis
- Trader segmentation
- Machine learning classification (predicting next-day sentiment)
- Trade pattern recognition (KMeans clustering)
- Interactive prediction GUI (Streamlit)

## Project Structure
- `trading_sentiment_analysis.py`: Core analysis and EDA
- `app.py`: ML model training and evaluation
- `gui.py`: Streamlit GUI for sentiment prediction
- `trade_pattern_recognition.py`: Trade pattern clustering
- `project_report.md`: Detailed project report
- `requirements.txt`: Python dependencies
- `Untitled.ipynb`: Example notebook for EDA and analysis

## Datasets & Model
**Download the required datasets and trained model from this Google Drive folder:**
[Project Datasets & Model](https://drive.google.com/drive/folders/1c4__1Fxcvr10JFzWfeAb6vXDSY6LHlpk?usp=drive_link)
- `historical_data.csv`
- `fear_greed_index.csv`
- `sentiment_classifier.pkl`

Place these files in your project directory before running the scripts.

## Usage
1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Run EDA and analysis:**
   ```sh
   python trading_sentiment_analysis.py
   ```
3. **Train the ML model:**
   ```sh
   python app.py
   ```
   The trained model will be saved as `sentiment_classifier.pkl` (or download from the Drive link above).
4. **Launch the prediction GUI:**
   ```sh
   streamlit run gui.py
   ```
5. **Run trade pattern recognition:**
   ```sh
   python trade_pattern_recognition.py
   ```

## Key Insights
- Sentiment strongly influences trading outcomes (PnL, win rate, trade size).
- Trader segmentation and clustering reveal distinct trading behaviors.
- The ML model can predict next-day sentiment to inform trading strategy.
- Pattern recognition helps identify and monitor trading styles for further analysis or risk management.

## Credits
- Data and code by Mithlesh Yadav.
- For questions, see the project report or contact the author.
