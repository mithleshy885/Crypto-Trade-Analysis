import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TradingSentimentAnalyzer:
    SENTIMENT_ORDER = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
    SENTIMENT_COLORS = ['#d62728', '#ff7f0e', '#bab0ac', '#2ca02c', '#1f77b4']
    def __init__(self):
        self.hist_df = None
        self.sentiment_df = None
        self.merged_df = None
        
    def load_data(self, hist_path, sentiment_path):
        """Load and basic preprocessing of datasets"""
        print("Loading datasets...")
        
        # Load historical trading data
        self.hist_df = pd.read_csv(hist_path)
        
        # Load sentiment data
        self.sentiment_df = pd.read_csv(sentiment_path)
        
        print(f"Historical data shape: {self.hist_df.shape}")
        print(f"Sentiment data shape: {self.sentiment_df.shape}")
        
    def preprocess_data(self):
        """Clean and prepare data for analysis"""
        print("Preprocessing data...")
        
        # Convert timestamps
        self.hist_df['datetime'] = pd.to_datetime(self.hist_df['Timestamp IST'],format='%d-%m-%Y %H:%M',dayfirst=True)
        self.hist_df['date'] = self.hist_df['datetime'].dt.date
        
        # Convert sentiment timestamps
        if 'timestamp' in self.sentiment_df.columns:
            self.sentiment_df['datetime'] = pd.to_datetime(self.sentiment_df['timestamp'], unit='s')
        else:
            self.sentiment_df['datetime'] = pd.to_datetime(self.sentiment_df['date'])
        
        self.sentiment_df['date'] = self.sentiment_df['datetime'].dt.date
        
        # Clean data
        self.hist_df = self.hist_df.dropna(subset=['Closed PnL'])
        
        # Create additional features
        self.hist_df['trade_hour'] = self.hist_df['datetime'].dt.hour
        self.hist_df['is_profitable'] = self.hist_df['Closed PnL'] > 0
        self.hist_df['abs_pnl'] = abs(self.hist_df['Closed PnL'])
        
        print("Data preprocessing completed!")
        
    def merge_datasets(self):
        """Merge trading data with sentiment data"""
        print("Merging datasets...")
        
        # Merge on date
        self.merged_df = self.hist_df.merge(
            self.sentiment_df[['date', 'classification', 'value']], 
            on='date', 
            how='inner'
        )
        

        print(f"Merged dataset shape: {self.merged_df.shape}")
        print(f"Date range: {self.merged_df['date'].min()} to {self.merged_df['date'].max()}")
        
    def exploratory_analysis(self):
        """Comprehensive exploratory data analysis"""
        print("Performing exploratory analysis...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Exploratory Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Sentiment distribution
        sentiment_counts = self.sentiment_df['classification'].value_counts().reindex(self.SENTIMENT_ORDER, fill_value=0)
        axes[0,0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=self.SENTIMENT_COLORS)
        axes[0,0].set_title('Market Sentiment Distribution')
        
        # 2. PnL distribution by sentiment
        sns.boxplot(data=self.merged_df, x='classification', y='Closed PnL', ax=axes[0,1], order=self.SENTIMENT_ORDER, palette=self.SENTIMENT_COLORS)
        axes[0,1].set_title('PnL Distribution by Sentiment')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Trade frequency by sentiment
        trade_freq = self.merged_df.groupby(['date', 'classification']).size().reset_index(name='trade_count')
        sns.boxplot(data=trade_freq, x='classification', y='trade_count', ax=axes[0,2], order=self.SENTIMENT_ORDER, palette=self.SENTIMENT_COLORS)
        axes[0,2].set_title('Daily Trade Frequency by Sentiment')
        axes[0,2].tick_params(axis='x', rotation=45)
        
        # 4. Win rate by sentiment
        win_rate = self.merged_df.groupby('classification')['is_profitable'].mean().reindex(self.SENTIMENT_ORDER)
        axes[1,0].bar(win_rate.index, win_rate.values, color=self.SENTIMENT_COLORS)
        axes[1,0].set_title('Win Rate by Sentiment')
        axes[1,0].set_ylabel('Win Rate')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 5. Average trade size by sentiment
        avg_size = self.merged_df.groupby('classification')['Size USD'].mean().reindex(self.SENTIMENT_ORDER)
        axes[1,1].bar(avg_size.index, avg_size.values, color=self.SENTIMENT_COLORS)
        axes[1,1].set_title('Average Trade Size (USD) by Sentiment')
        axes[1,1].set_ylabel('Average Size USD')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # 6. Side preference by sentiment
        side_sentiment = pd.crosstab(self.merged_df['classification'], self.merged_df['Side'], normalize='index').reindex(self.SENTIMENT_ORDER)
        side_sentiment.plot(kind='bar', ax=axes[1,2], color=['#8c564b', '#e377c2'])
        axes[1,2].set_title('Trading Side Preference by Sentiment')
        axes[1,2].set_ylabel('Proportion')
        axes[1,2].tick_params(axis='x', rotation=45)
        axes[1,2].legend(title='Side')
        
        plt.tight_layout()
        plt.show()
        
    def sentiment_performance_analysis(self):
        """Detailed analysis of performance by sentiment"""
        print("Analyzing sentiment-performance relationship...")
        
        # Create performance metrics by sentiment
        performance_stats = self.merged_df.groupby('classification').agg({
            'Closed PnL': ['mean', 'median', 'std', 'sum'],
            'Size USD': ['mean', 'median'],
            'is_profitable': 'mean',
            'Fee': 'mean',
            'Account': 'nunique'
        }).round(4)
        performance_stats = performance_stats.reindex(self.SENTIMENT_ORDER)
        performance_stats.columns = ['_'.join(col).strip() for col in performance_stats.columns]
        print("\nPerformance Statistics by Sentiment:")
        print(performance_stats)
        
        # Visualize key performance metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Analysis by Market Sentiment', fontsize=16, fontweight='bold')
        
        # Average PnL
        avg_pnl = self.merged_df.groupby('classification')['Closed PnL'].mean().reindex(self.SENTIMENT_ORDER)
        axes[0,0].bar(avg_pnl.index, avg_pnl.values, color=self.SENTIMENT_COLORS)
        axes[0,0].set_title('Average PnL by Sentiment')
        axes[0,0].set_ylabel('Average PnL')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Total PnL
        total_pnl = self.merged_df.groupby('classification')['Closed PnL'].sum().reindex(self.SENTIMENT_ORDER)
        axes[0,1].bar(total_pnl.index, total_pnl.values, color=self.SENTIMENT_COLORS)
        axes[0,1].set_title('Total PnL by Sentiment')
        axes[0,1].set_ylabel('Total PnL')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Win Rate
        win_rate = self.merged_df.groupby('classification')['is_profitable'].mean().reindex(self.SENTIMENT_ORDER)
        axes[1,0].bar(win_rate.index, win_rate.values, color=self.SENTIMENT_COLORS)
        axes[1,0].set_title('Win Rate by Sentiment')
        axes[1,0].set_ylabel('Win Rate')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Risk-Reward Analysis
        profit_trades = self.merged_df[self.merged_df['is_profitable'] == True]
        loss_trades = self.merged_df[self.merged_df['is_profitable'] == False]
        
        avg_profit = profit_trades.groupby('classification')['Closed PnL'].mean().reindex(self.SENTIMENT_ORDER)
        avg_loss = loss_trades.groupby('classification')['Closed PnL'].mean().reindex(self.SENTIMENT_ORDER)
        risk_reward = avg_profit / abs(avg_loss)
        
        axes[1,1].bar(risk_reward.index, risk_reward.values, color=self.SENTIMENT_COLORS)
        axes[1,1].set_title('Risk-Reward Ratio by Sentiment')
        axes[1,1].set_ylabel('Risk-Reward Ratio')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return performance_stats
    
    def trader_segmentation_analysis(self):
        """Analyze different trader segments and their sentiment responsiveness"""
        print("Performing trader segmentation analysis...")
        
        # Create trader performance metrics
        trader_stats = self.merged_df.groupby('Account').agg({
            'Closed PnL': ['sum', 'mean', 'count', 'std'],
            'Size USD': 'mean',
            'is_profitable': 'mean',
            'Fee': 'sum'
        }).round(4)
        
        trader_stats.columns = ['_'.join(col).strip() for col in trader_stats.columns]
        trader_stats = trader_stats.rename(columns={
            'Closed PnL_sum': 'total_pnl',
            'Closed PnL_mean': 'avg_pnl',
            'Closed PnL_count': 'trade_count',
            'Closed PnL_std': 'pnl_volatility',
            'Size USD_mean': 'avg_trade_size',
            'is_profitable_mean': 'win_rate',
            'Fee_sum': 'total_fees'
        })
        
        # Segment traders based on performance
        trader_stats['performance_category'] = pd.cut(
            trader_stats['total_pnl'], 
            bins=[-np.inf, -1000, 0, 1000, np.inf], 
            labels=['High Loss', 'Small Loss', 'Small Profit', 'High Profit']
        )
        
        # Activity level segmentation
        trader_stats['activity_level'] = pd.cut(
            trader_stats['trade_count'], 
            bins=[0, 10, 50, 200, np.inf], 
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # Analyze sentiment responsiveness by trader segment
        merged_with_segments = self.merged_df.merge(
            trader_stats[['performance_category', 'activity_level']], 
            left_on='Account', 
            right_index=True
        )
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Trader Segmentation Analysis', fontsize=16, fontweight='bold')
        
        # Performance category distribution
        trader_stats['performance_category'].value_counts().plot(kind='bar', ax=axes[0,0], color='#8c564b')
        axes[0,0].set_title('Trader Performance Categories')
        axes[0,0].set_ylabel('Number of Traders')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Activity level distribution
        trader_stats['activity_level'].value_counts().plot(kind='bar', ax=axes[0,1], color='#e377c2')
        axes[0,1].set_title('Trader Activity Levels')
        axes[0,1].set_ylabel('Number of Traders')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Sentiment response by performance category
        sentiment_response = merged_with_segments.groupby(['performance_category', 'classification'])['Closed PnL'].mean().unstack().reindex(columns=self.SENTIMENT_ORDER)
        sentiment_response.plot(kind='bar', ax=axes[1,0], color=self.SENTIMENT_COLORS)
        axes[1,0].set_title('Average PnL by Performance Category & Sentiment')
        axes[1,0].set_ylabel('Average PnL')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].legend(title='Sentiment')
        
        # Trading frequency by sentiment and activity level
        freq_analysis = merged_with_segments.groupby(['activity_level', 'classification']).size().unstack().reindex(columns=self.SENTIMENT_ORDER)
        freq_analysis.plot(kind='bar', ax=axes[1,1], color=self.SENTIMENT_COLORS)
        axes[1,1].set_title('Trade Frequency by Activity Level & Sentiment')
        axes[1,1].set_ylabel('Number of Trades')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].legend(title='Sentiment')
        
        plt.tight_layout()
        plt.show()
        
        return trader_stats, merged_with_segments
    
    def temporal_pattern_analysis(self):
        """Analyze temporal patterns and sentiment transitions"""
        print("Analyzing temporal patterns...")
        # Daily aggregations
        daily_stats = self.merged_df.groupby(['date', 'classification']).agg({
            'Closed PnL': ['sum', 'mean'],
            'Size USD': 'sum',
            'Account': 'nunique',
            'is_profitable': 'mean'
        }).round(4)
        daily_stats.columns = ['_'.join(col).strip() for col in daily_stats.columns]
        daily_stats = daily_stats.reset_index()
        # Time series visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Temporal Analysis', fontsize=16, fontweight='bold')
        # Daily PnL by sentiment
        for i, sentiment in enumerate(self.SENTIMENT_ORDER):
            sentiment_data = daily_stats[daily_stats['classification'] == sentiment]
            axes[0,0].plot(sentiment_data['date'], sentiment_data['Closed PnL_sum'], 
                          label=sentiment, marker='o', markersize=3, color=self.SENTIMENT_COLORS[i])
        axes[0,0].set_title('Daily Total PnL by Sentiment')
        axes[0,0].set_ylabel('Total PnL')
        axes[0,0].legend()
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Trading volume by sentiment
        for i, sentiment in enumerate(self.SENTIMENT_ORDER):
            sentiment_data = daily_stats[daily_stats['classification'] == sentiment]
            axes[0,1].plot(sentiment_data['date'], sentiment_data['Size USD_sum'], 
                          label=sentiment, marker='o', markersize=3, color=self.SENTIMENT_COLORS[i])
        axes[0,1].set_title('Daily Trading Volume by Sentiment')
        axes[0,1].set_ylabel('Total Volume (USD)')
        axes[0,1].legend()
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Active traders by sentiment
        for i, sentiment in enumerate(self.SENTIMENT_ORDER):
            sentiment_data = daily_stats[daily_stats['classification'] == sentiment]
            axes[1,0].plot(sentiment_data['date'], sentiment_data['Account_nunique'], 
                          label=sentiment, marker='o', markersize=3, color=self.SENTIMENT_COLORS[i])
        axes[1,0].set_title('Daily Active Traders by Sentiment')
        axes[1,0].set_ylabel('Number of Active Traders')
        axes[1,0].legend()
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Win rate over time
        for i, sentiment in enumerate(self.SENTIMENT_ORDER):
            sentiment_data = daily_stats[daily_stats['classification'] == sentiment]
            axes[1,1].plot(sentiment_data['date'], sentiment_data['is_profitable_mean'], 
                          label=sentiment, marker='o', markersize=3, color=self.SENTIMENT_COLORS[i])
        axes[1,1].set_title('Daily Win Rate by Sentiment')
        axes[1,1].set_ylabel('Win Rate')
        axes[1,1].legend()
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return daily_stats
    
    def advanced_insights(self):
        """Generate advanced insights and trading signals"""
        print("Generating advanced insights...")
        
        insights = {}
        
        # 1. Contrarian vs Momentum traders (using Extreme Fear vs Extreme Greed)
        trader_sentiment_behavior = self.merged_df.groupby(['Account', 'classification']).agg({
            'Closed PnL': 'mean',
            'Size USD': 'mean',
            'Side': lambda x: (x == 'buy').mean()  # Proportion of buy orders
        }).reset_index()
        
        # Pivot to compare behavior across sentiments
        pnl_pivot = trader_sentiment_behavior.pivot(index='Account', columns='classification', values='Closed PnL')
        buy_ratio_pivot = trader_sentiment_behavior.pivot(index='Account', columns='classification', values='Side')
        
        # Identify contrarian traders (profit more during fear, buy more during fear)
        if 'Extreme Fear' in pnl_pivot.columns and 'Extreme Greed' in pnl_pivot.columns:
            contrarian_score = (pnl_pivot['Extreme Fear'] - pnl_pivot['Extreme Greed']) + (buy_ratio_pivot['Extreme Fear'] - buy_ratio_pivot['Extreme Greed'])
            insights['contrarian_traders'] = contrarian_score.nlargest(10)
            insights['momentum_traders'] = contrarian_score.nsmallest(10)
        
        # 2. Best performing strategies by sentiment
        strategy_performance = self.merged_df.groupby(['classification', 'Side']).agg({
            'Closed PnL': ['mean', 'sum', 'count'],
            'is_profitable': 'mean'
        }).round(4)
        
        insights['strategy_performance'] = strategy_performance
        
        # 3. Optimal trade sizing by sentiment
        size_bins = pd.qcut(self.merged_df['Size USD'], q=5, labels=['XS', 'S', 'M', 'L', 'XL'])
        size_performance = self.merged_df.groupby(['classification', size_bins]).agg({
            'Closed PnL': 'mean',
            'is_profitable': 'mean'
        }).round(4)
        
        insights['size_performance'] = size_performance
        
        # 4. Market timing insights
        hour_performance = self.merged_df.groupby(['classification', 'trade_hour']).agg({
            'Closed PnL': 'mean',
            'is_profitable': 'mean',
            'Size USD': 'mean'
        }).round(4)
        
        insights['hour_performance'] = hour_performance
        
        # Visualization of key insights
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Advanced Trading Insights', fontsize=16, fontweight='bold')
        
        # Strategy performance heatmap
        strategy_pnl = strategy_performance['Closed PnL']['mean'].unstack().reindex(self.SENTIMENT_ORDER)
        sns.heatmap(strategy_pnl, annot=True, cmap='RdYlGn', center=0, ax=axes[0,0])
        axes[0,0].set_title('Average PnL by Strategy & Sentiment')
        
        # Size performance heatmap
        size_pnl = size_performance['Closed PnL'].unstack().reindex(self.SENTIMENT_ORDER)
        sns.heatmap(size_pnl, annot=True, cmap='RdYlGn', center=0, ax=axes[0,1])
        axes[0,1].set_title('Average PnL by Trade Size & Sentiment')
        
        # Hourly performance (plot all five sentiments)
        for i, sentiment in enumerate(self.SENTIMENT_ORDER):
            if sentiment in hour_performance.index.get_level_values(0):
                hourly = hour_performance.loc[sentiment]['Closed PnL']
                axes[1,0].plot(hourly.index, hourly.values, marker='o', label=sentiment, color=self.SENTIMENT_COLORS[i])
        axes[1,0].set_title('Hourly Performance by Sentiment')
        axes[1,0].set_xlabel('Hour of Day')
        axes[1,0].set_ylabel('Average PnL')
        axes[1,0].legend()
        
        # Top contrarian traders (if available)
        if 'contrarian_traders' in insights:
            top_contrarians = insights['contrarian_traders'].head(10)
            axes[1,1].barh(range(len(top_contrarians)), top_contrarians.values, color='#9467bd')
            axes[1,1].set_title('Top Contrarian Traders')
            axes[1,1].set_xlabel('Contrarian Score')
            axes[1,1].set_yticks(range(len(top_contrarians)))
            axes[1,1].set_yticklabels([f'Trader {i+1}' for i in range(len(top_contrarians))])
        
        plt.tight_layout()
        plt.show()
        
        return insights
    
    def generate_recommendations(self, insights):
        """Generate actionable trading recommendations"""
        print("\n" + "="*60)
        print("TRADING STRATEGY RECOMMENDATIONS")
        print("="*60)
        
        recommendations = []
        
        # Performance by sentiment
        sentiment_performance = self.merged_df.groupby('classification')['Closed PnL'].agg(['mean', 'sum']).reindex(self.SENTIMENT_ORDER)
        best_sentiment = sentiment_performance['mean'].idxmax()
        
        recommendations.append(f"1. SENTIMENT TIMING: Focus trading during {best_sentiment} periods - "
                             f"average PnL is {sentiment_performance.loc[best_sentiment, 'mean']:.2f}")
        
        # Best trading side by sentiment
        if 'strategy_performance' in insights:
            strategy_perf = insights['strategy_performance']['Closed PnL']['mean']
            for sentiment in self.SENTIMENT_ORDER:
                if sentiment in strategy_perf.index:
                    best_side = strategy_perf.loc[sentiment].idxmax()
                    recommendations.append(f"2. DIRECTIONAL BIAS: During {sentiment}, prefer {best_side} trades - "
                                         f"average PnL: {strategy_perf.loc[sentiment, best_side]:.2f}")
        
        # Optimal trade sizes
        if 'size_performance' in insights:
            size_perf = insights['size_performance']['Closed PnL']
            for sentiment in self.SENTIMENT_ORDER:
                if sentiment in size_perf.index:
                    best_size = size_perf.loc[sentiment].idxmax()
                    recommendations.append(f"3. POSITION SIZING: During {sentiment}, use {best_size} positions for "
                                         f"optimal performance: {size_perf.loc[sentiment, best_size]:.2f}")
        
        # Risk management insights
        win_rates = self.merged_df.groupby('classification')['is_profitable'].mean().reindex(self.SENTIMENT_ORDER)
        for sentiment, win_rate in win_rates.items():
            if pd.isna(win_rate):
                continue
            risk_level = "HIGH" if win_rate < 0.4 else "MEDIUM" if win_rate < 0.6 else "LOW"
            recommendations.append(f"4. RISK MANAGEMENT: {sentiment} periods have {win_rate:.1%} win rate - "
                                 f"{risk_level} risk environment")
        
        # Market timing
        if 'hour_performance' in insights:
            hour_perf = insights['hour_performance']['Closed PnL']
            for sentiment in self.SENTIMENT_ORDER:
                if sentiment in hour_perf.index:
                    best_hours = hour_perf.loc[sentiment].nlargest(3)
                    recommendations.append(f"5. MARKET TIMING: Best hours for {sentiment} trading: "
                                         f"{', '.join(map(str, best_hours.index.tolist()))}")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{rec}")
            
        print("\n" + "="*60)
        print("KEY INSIGHTS SUMMARY")
        print("="*60)
        
        # Summary statistics
        total_trades = len(self.merged_df)
        total_pnl = self.merged_df['Closed PnL'].sum()
        overall_win_rate = self.merged_df['is_profitable'].mean()
        
        print(f"• Total Trades Analyzed: {total_trades:,}")
        print(f"• Total PnL: ${total_pnl:,.2f}")
        print(f"• Overall Win Rate: {overall_win_rate:.1%}")
        print(f"• Analysis Period: {self.merged_df['date'].min()} to {self.merged_df['date'].max()}")
        print(f"• Unique Traders: {self.merged_df['Account'].nunique():,}")
        
        return recommendations
    
    def run_complete_analysis(self, hist_path, sentiment_path):
        """Run the complete analysis pipeline"""
        print("Starting comprehensive trading sentiment analysis...")
        print("="*60)
        
        # Load and preprocess data
        self.load_data(hist_path, sentiment_path)
        self.preprocess_data()
        self.merge_datasets()
        
        # Run all analyses
        self.exploratory_analysis()
        performance_stats = self.sentiment_performance_analysis()
        trader_stats, segmented_data = self.trader_segmentation_analysis()
        daily_stats = self.temporal_pattern_analysis()
        insights = self.advanced_insights()
        
        # Generate recommendations
        recommendations = self.generate_recommendations(insights)
        
        print("\nAnalysis completed successfully!")
        return {
            'performance_stats': performance_stats,
            'trader_stats': trader_stats,
            'daily_stats': daily_stats,
            'insights': insights,
            'recommendations': recommendations
        }

# Usage Example:
# analyzer = TradingSentimentAnalyzer()
# results = analyzer.run_complete_analysis('historical_data.csv', 'fear_greed_index.csv')

# For individual analyses, you can also run:
# analyzer.load_data('historical_data.csv', 'fear_greed_index.csv')
# analyzer.preprocess_data()
# analyzer.merge_datasets()
# analyzer.exploratory_analysis()
# performance = analyzer.sentiment_performance_analysis()
# traders, segments = analyzer.trader_segmentation_analysis()
# temporal = analyzer.temporal_pattern_analysis()
# insights = analyzer.advanced_insights()

# recommendations = analyzer.generate_recommendations(insights)