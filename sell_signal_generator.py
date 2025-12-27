# =============================================================================
# 매도 신호 생성 모듈
# =============================================================================
import pandas as pd
import numpy as np
from technical_indicators import TechnicalIndicators

class SellSignalGenerator:
    """매도 신호 생성 클래스"""
    
    def __init__(self, data):
        """
        Args:
            data: OHLCV 데이터프레임 (기술적 지표는 동적으로 계산)
        """
        self.data = data.copy()
        # 기본 기술적 지표는 이미 계산되어 있을 수 있지만, 동적 계산을 위해 원본 데이터 유지
    
    def generate_rsi_signal(self, rsi_overbought=70, rsi_period=14, weight=1.0):
        """RSI 기반 매도 신호 (하향돌파 방식) - 동적 계산"""
        # RSI 동적 계산
        indicators = TechnicalIndicators(self.data)
        rsi = indicators.calculate_rsi(period=rsi_period)
        
        # 이전일과 현재일 비교
        rsi_prev = rsi.shift(1)
        
        # 하향돌파: 이전일 > overbought, 현재일 <= overbought
        signal = np.where(
            (rsi_prev > rsi_overbought) & (rsi <= rsi_overbought),
            weight,  # 하향돌파 시 고정 신호 강도
            0
        )
        return pd.Series(signal, index=self.data.index)
    
    def generate_stochastic_signal(self, k_period=14, d_period=3, slow_period=3, overbought=80, weight=1.0):
        """스토캐스틱 기반 매도 신호 - 동적 계산"""
        # 스토캐스틱 동적 계산
        indicators = TechnicalIndicators(self.data)
        k_percent, d_percent = indicators.calculate_stochastic(
            k_period=k_period, 
            d_period=d_period, 
            slow_period=slow_period
        )
        
        # K와 D가 모두 과매수 구간을 초과하면 신호 생성
        signal = np.where(
            (k_percent > overbought) & (d_percent > overbought),
            np.minimum(
                ((k_percent - overbought) / (100 - overbought) + 
                 (d_percent - overbought) / (100 - overbought)) / 2,
                1.0
            ) * weight,
            0
        )
        return pd.Series(signal, index=self.data.index)
    
    def generate_obv_signal(self, obv_period=20, weight=1.0):
        """OBV 기반 매도 신호 (OBV가 하락 추세일 때)"""
        # OBV 계산
        indicators = TechnicalIndicators(self.data)
        obv = indicators.calculate_obv()
        
        # OBV 이동평균과 비교
        obv_ma = obv.rolling(window=obv_period).mean()
        obv_signal = np.where(
            obv < obv_ma,
            np.minimum((obv_ma - obv) / (obv_ma + 1e-10), 1.0) * weight,
            0
        )
        return pd.Series(obv_signal, index=self.data.index)
    
    def generate_ema_signal(self, ema_period=60):
        """EMA 기반 매도 신호 (20일선이 ema_period일선을 하향돌파)"""
        # EMA 계산 (20일 고정, ema_period일)
        indicators = TechnicalIndicators(self.data)
        ema_20 = indicators.calculate_ema(period=20)
        ema_long = indicators.calculate_ema(period=ema_period)
        
        # 이전일: EMA_20 > EMA_long, 현재일: EMA_20 <= EMA_long (하향돌파)
        ema_20_prev = ema_20.shift(1)
        ema_long_prev = ema_long.shift(1)
        
        signal = np.where(
            (ema_20_prev > ema_long_prev) & (ema_20 <= ema_long),
            1.0,  # 하향돌파 시 고정 신호 강도 (weight 없음)
            0
        )
        return pd.Series(signal, index=self.data.index)
    
    def generate_atr_signal(self, atr_multiplier=2.0, weight=1.0):
        """ATR 기반 매도 신호 (변동성이 클 때)"""
        # ATR 계산
        indicators = TechnicalIndicators(self.data)
        atr = indicators.calculate_atr(period=14)
        
        # ATR 이동평균 대비 현재 ATR이 높으면 신호
        atr_ma = atr.rolling(window=20).mean()
        signal = np.where(
            atr > atr_ma * atr_multiplier,
            np.minimum((atr - atr_ma * atr_multiplier) / (atr_ma * atr_multiplier + 1e-10), 1.0) * weight,
            0
        )
        return pd.Series(signal, index=self.data.index)
    
    def generate_bollinger_signal(self, bb_position_threshold=100, bb_period=20, bb_std_dev=2.0, weight=1.0):
        """볼린저밴드 기반 매도 신호 (상단 밴드에 정확히 닿을 때 매도)"""
        # 볼린저밴드 계산
        indicators = TechnicalIndicators(self.data)
        _, _, _, bb_position = indicators.calculate_bollinger_bands(period=bb_period, std_dev=bb_std_dev)
        
        # BB_Position이 임계값 이상이면 신호 생성
        signal = np.where(
            bb_position >= bb_position_threshold,
            np.minimum((bb_position - bb_position_threshold) / (100 - bb_position_threshold + 1e-10), 1.0) * weight,
            0
        )
        return pd.Series(signal, index=self.data.index)
    
    def generate_combined_signal(self, params):
        """
        모든 지표를 결합한 종합 기술점수 생성
        
        Args:
            params: 파라미터 딕셔너리
                - rsi_overbought, rsi_period, rsi_weight
                - obv_period, obv_weight
                - atr_multiplier, atr_weight
                - bb_position_threshold, bb_period, bb_std_dev, bb_weight
        
        Returns:
            종합 기술점수 (0 이상의 값)
        """
        # 각 지표별 신호 생성
        rsi_signal = self.generate_rsi_signal(
            params.get('rsi_overbought', 70),
            params.get('rsi_period', 14),
            params.get('rsi_weight', 1.0)
        )
        
        obv_signal = self.generate_obv_signal(
            params.get('obv_period', 20),
            params.get('obv_weight', 1.0)
        )
        
        atr_signal = self.generate_atr_signal(
            params.get('atr_multiplier', 2.0),
            params.get('atr_weight', 1.0)
        )
        
        # 볼린저밴드 신호
        bb_signal = self.generate_bollinger_signal(
            params.get('bb_position_threshold', 100),
            params.get('bb_period', 20),
            params.get('bb_std_dev', 2.0),
            params.get('bb_weight', 1.0)
        )
        
        # 가중합으로 종합 기술점수 계산 (EMA 제외)
        combined_signal = (
            rsi_signal +
            obv_signal +
            atr_signal +
            bb_signal
        )
        
        # pandas Series로 변환하여 인덱스 유지
        if isinstance(combined_signal, np.ndarray):
            combined_signal = pd.Series(combined_signal, index=self.data.index)
        
        return combined_signal
    
    def normalize_signal(self, signal):
        """신호를 0-1 범위로 정규화 (선택적)"""
        if signal.max() > 0:
            return signal / signal.max()
        return signal

