import streamlit as st
import numpy as np
import joblib

# Load the trained model
clf = joblib.load('sentiment_classifier.pkl')

st.title('Crypto Sentiment Prediction')
st.write('Enter today\'s trading features to predict if tomorrow will be a Greed or Fear day!')

# Input fields for features
def user_input_features():
    closed_pnl = st.number_input('Closed PnL', value=0.0)
    size_usd = st.number_input('Trade Size (USD)', value=100.0)
    fee = st.number_input('Fee', value=0.0)
    trade_hour = st.slider('Trade Hour', 0, 23, 12)
    is_profitable = st.selectbox('Is Profitable?', [0, 1])
    abs_pnl = st.number_input('Absolute PnL', value=0.0)
    pnl_rolling_mean_3 = st.number_input('PnL Rolling Mean (3)', value=0.0)
    pnl_rolling_std_3 = st.number_input('PnL Rolling Std (3)', value=0.0)
    size_rolling_mean_3 = st.number_input('Size Rolling Mean (3)', value=100.0)
    size_rolling_std_3 = st.number_input('Size Rolling Std (3)', value=0.0)
    sentiment_lag1 = st.selectbox('Today\'s Sentiment', [
        'Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'
    ])
    sentiment_map = {'Extreme Fear': 0, 'Fear': 1, 'Neutral': 2, 'Greed': 3, 'Extreme Greed': 4}
    sentiment_lag1_val = sentiment_map[sentiment_lag1]
    features = np.array([
        closed_pnl, size_usd, fee, trade_hour, is_profitable, abs_pnl,
        pnl_rolling_mean_3, pnl_rolling_std_3, size_rolling_mean_3, size_rolling_std_3,
        sentiment_lag1_val
    ]).reshape(1, -1)
    return features

input_features = user_input_features()

if st.button('Predict Tomorrow\'s Sentiment'):
    pred = clf.predict(input_features)[0]
    st.subheader(f"Prediction: {pred}")
    if pred in ['Greed', 'Extreme Greed']:
        st.success('Tomorrow is likely to be a Greed day!')
    elif pred in ['Fear', 'Extreme Fear']:
        st.warning('Tomorrow is likely to be a Fear day!')
    else:
        st.info('Tomorrow is likely to be Neutral.')
