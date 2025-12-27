# =============================================================================
# 기술적 지표 계산 모듈
# =============================================================================
import pandas as pd
import numpy as np

class TechnicalIndicators:
    """기술적 지표 계산 클래스"""
    
    def __init__(self, data):
        """
        Args:
            data: OHLCV 데이터프레임 (인덱스는 날짜)
        """
        self.data = data.copy()
        self.calculate_all_indicators()
    
    def calculate_rsi(self, period=14):
        """RSI (Relative Strength Index) 계산"""
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_stochastic(self, k_period=14, d_period=3, slow_period=3):
        """스토캐스틱 오실레이터 계산"""
        low_min = self.data['Low'].rolling(window=k_period).min()
        high_max = self.data['High'].rolling(window=k_period).max()
        
        k_percent = 100 * ((self.data['Close'] - low_min) / (high_max - low_min))
        # D는 K의 이동평균, slow_period는 D를 계산할 때 사용
        k_smoothed = k_percent.rolling(window=d_period).mean()
        d_percent = k_smoothed.rolling(window=slow_period).mean()
        
        return k_percent, d_percent
    
    def calculate_obv(self):
        """OBV (On-Balance Volume) 계산"""
        obv = pd.Series(index=self.data.index, dtype=float)
        obv.iloc[0] = self.data['Volume'].iloc[0]
        
        for i in range(1, len(self.data)):
            if self.data['Close'].iloc[i] > self.data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + self.data['Volume'].iloc[i]
            elif self.data['Close'].iloc[i] < self.data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - self.data['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def calculate_bollinger_bands(self, period=20, std_dev=2):
        """볼린저 밴드 계산"""
        sma = self.data['Close'].rolling(window=period).mean()
        std = self.data['Close'].rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        # 볼린저 밴드 위치 (0-100, 상단에 가까울수록 높은 값)
        bb_position = ((self.data['Close'] - lower_band) / (upper_band - lower_band)) * 100
        
        return sma, upper_band, lower_band, bb_position
    
    def calculate_ema(self, period=20):
        """EMA (Exponential Moving Average) 계산"""
        return self.data['Close'].ewm(span=period, adjust=False).mean()
    
    def calculate_atr(self, period=14):
        """ATR (Average True Range) 계산"""
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def calculate_all_indicators(self):
        """모든 기술적 지표 계산"""
        # RSI
        self.data['RSI'] = self.calculate_rsi(period=14)
        
        # 스토캐스틱
        k_percent, d_percent = self.calculate_stochastic(k_period=14, d_period=3)
        self.data['Stoch_K'] = k_percent
        self.data['Stoch_D'] = d_percent
        
        # OBV
        self.data['OBV'] = self.calculate_obv()
        
        # 볼린저 밴드
        sma, upper_band, lower_band, bb_position = self.calculate_bollinger_bands(period=20, std_dev=2)
        self.data['BB_Middle'] = sma
        self.data['BB_Upper'] = upper_band
        self.data['BB_Lower'] = lower_band
        self.data['BB_Position'] = bb_position
        
        # EMA (20일, 60일)
        self.data['EMA_20'] = self.calculate_ema(period=20)
        self.data['EMA_60'] = self.calculate_ema(period=60)
        
        # ATR
        self.data['ATR'] = self.calculate_atr(period=14)
    
    def get_data(self):
        """계산된 지표가 포함된 데이터 반환"""
        return self.data.copy()

