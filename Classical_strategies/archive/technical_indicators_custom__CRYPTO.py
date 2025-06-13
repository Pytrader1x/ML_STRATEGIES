"""
Technical Indicators Custom - Required indicators for crypto strategy
"""

import pandas as pd
import numpy as np


class TIC:
    """Technical Indicators Custom class"""
    
    @staticmethod
    def add_neuro_trend_intelligent(df):
        """Add Neuro Trend Intelligent indicator"""
        # Simple trend direction based on multiple EMAs
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
        
        # Direction: 1 for bullish, -1 for bearish, 0 for neutral
        df['NTI_Direction'] = 0
        
        # Bullish when price > EMA20 > EMA50 > EMA100
        bullish = (df['Close'] > df['EMA_20']) & (df['EMA_20'] > df['EMA_50']) & (df['EMA_50'] > df['EMA_100'])
        df.loc[bullish, 'NTI_Direction'] = 1
        
        # Bearish when price < EMA20 < EMA50 < EMA100
        bearish = (df['Close'] < df['EMA_20']) & (df['EMA_20'] < df['EMA_50']) & (df['EMA_50'] < df['EMA_100'])
        df.loc[bearish, 'NTI_Direction'] = -1
        
        # Clean up temporary columns
        df.drop(['EMA_20', 'EMA_50', 'EMA_100'], axis=1, inplace=True)
        
        return df
    
    @staticmethod
    def add_market_bias(df):
        """Add Market Bias indicator"""
        # Market bias based on price position relative to moving averages
        ma_50 = df['Close'].rolling(window=50).mean()
        ma_200 = df['Close'].rolling(window=200).mean()
        
        # Calculate bias
        df['MB_Bias'] = 0
        
        # Strong bullish bias
        strong_bull = (df['Close'] > ma_50) & (ma_50 > ma_200) & (df['Close'] > ma_200 * 1.05)
        df.loc[strong_bull, 'MB_Bias'] = 2
        
        # Moderate bullish bias
        mod_bull = (df['Close'] > ma_50) & (ma_50 > ma_200) & ~strong_bull
        df.loc[mod_bull, 'MB_Bias'] = 1
        
        # Strong bearish bias
        strong_bear = (df['Close'] < ma_50) & (ma_50 < ma_200) & (df['Close'] < ma_200 * 0.95)
        df.loc[strong_bear, 'MB_Bias'] = -2
        
        # Moderate bearish bias
        mod_bear = (df['Close'] < ma_50) & (ma_50 < ma_200) & ~strong_bear
        df.loc[mod_bear, 'MB_Bias'] = -1
        
        return df
    
    @staticmethod
    def add_intelligent_chop(df):
        """Add Intelligent Chop indicator (market choppiness)"""
        # ATR for volatility
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=14).mean()
        
        # Price movement efficiency
        period = 20
        net_change = np.abs(df['Close'] - df['Close'].shift(period))
        sum_changes = df['Close'].diff().abs().rolling(window=period).sum()
        
        efficiency = net_change / sum_changes
        efficiency = efficiency.fillna(0.5)
        
        # IC_Signal: 1 for trending, 0 for choppy
        df['IC_Signal'] = 0
        trending = efficiency > 0.3  # Market is trending if efficiency > 30%
        df.loc[trending, 'IC_Signal'] = 1
        
        # Add IC_Regime (1=Trending, 2=Range, 3=Chop)
        df['IC_Regime'] = 3  # Default to chop
        df.loc[efficiency > 0.3, 'IC_Regime'] = 1  # Trending
        df.loc[(efficiency > 0.15) & (efficiency <= 0.3), 'IC_Regime'] = 2  # Range
        
        # Add IC_RegimeName
        df['IC_RegimeName'] = 'Chop'
        df.loc[df['IC_Regime'] == 1, 'IC_RegimeName'] = 'Trend'
        df.loc[df['IC_Regime'] == 2, 'IC_RegimeName'] = 'Range'
        
        # Add IC_ATR_Normalized (normalized ATR)
        df['IC_ATR_Normalized'] = atr / df['Close'] * 100  # ATR as percentage of price
        
        return df