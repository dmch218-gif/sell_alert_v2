# =============================================================================
# 매도 신호 감지 모듈
# =============================================================================
import pandas as pd
import math
from datetime import datetime
from sell_signal_generator import SellSignalGenerator
from sell_position_manager import SellPositionManager

class SignalDetector:
    """매도 신호 감지 클래스"""
    
    def __init__(self, data: pd.DataFrame, params: dict, buy_date: str, buy_price: float = None, sell_count: int = 0):
        """
        Args:
            data: 기술적 지표가 포함된 데이터프레임
            params: 파라미터 딕셔너리
            buy_date: 매수일 (YYYY-MM-DD)
            buy_price: 매수 평균가 (None이면 매수일 종가 사용)
            sell_count: 신호 발생 거래일 수 (신호가중치 계산용)
        """
        self.data = data
        self.params = params
        self.buy_date = pd.to_datetime(buy_date)
        self.sell_count = sell_count  # 거래일 단위 신호 카운트
        
        # 매수 평균가 설정 (None이면 매수일 종가의 2배 사용)
        if buy_price is None:
            # 매수일(4/11) 종가의 2배
            buy_date_data = data[data['Date'] >= self.buy_date]
            if not buy_date_data.empty:
                buy_date_price = buy_date_data.iloc[0]['Close']  # 매수일 종가
                self.buy_price = buy_date_price * 2.0  # 매수일 종가의 2배 = 매수 평균가
            else:
                self.buy_price = data['Close'].iloc[0] * 2.0
        else:
            self.buy_price = buy_price
        
        # 신호 생성기 초기화
        self.signal_generator = SellSignalGenerator(data)
        
    def detect_signal(self) -> dict:
        """
        최신 데이터에서 매도 신호 감지
        
        Returns:
            신호 정보 딕셔너리
        """
        if self.data is None or self.data.empty:
            return None
        
        # 최신 데이터
        latest = self.data.iloc[-1]
        latest_date = latest['Date']
        latest_price = latest['Close']
        
        # 저점 대비 보유일수 계산 (최근 저점부터 현재까지)
        low_idx = self.data['Low'].idxmin()  # 최저점 인덱스
        low_date = self.data.loc[low_idx, 'Date']
        days_from_low = (latest_date - low_date).days
        if days_from_low < 0:
            days_from_low = 0
        
        # 매수일 기준 보유 일수 (시간 가중치 계산용)
        days_held = (latest_date - self.buy_date).days
        if days_held < 0:
            days_held = 0
        
        # 수익률 계산 (내부 계산용)
        current_return = (latest_price - self.buy_price) / self.buy_price
        
        # 개별 지표별 신호 계산
        rsi_signal = self.signal_generator.generate_rsi_signal(
            self.params.get('rsi_overbought', 70),
            self.params.get('rsi_period', 14),
            self.params.get('rsi_weight', 1.0)
        )
        
        obv_signal = self.signal_generator.generate_obv_signal(
            self.params.get('obv_period', 20),
            self.params.get('obv_weight', 1.0)
        )
        
        atr_signal = self.signal_generator.generate_atr_signal(
            self.params.get('atr_multiplier', 2.0),
            self.params.get('atr_weight', 1.0)
        )
        
        bb_signal = self.signal_generator.generate_bollinger_signal(
            self.params.get('bb_position_threshold', 100),
            self.params.get('bb_period', 20),
            self.params.get('bb_std_dev', 2.0),
            self.params.get('bb_weight', 1.0)
        )
        
        # 통합 신호 계산 (params 딕셔너리 전달)
        combined_signal = self.signal_generator.generate_combined_signal(self.params)
        
        # 최신 신호 강도
        signal_strength = combined_signal.iloc[-1] if not combined_signal.empty else 0
        
        # 신호 발생 지표 확인
        signal_indicators = []
        if not rsi_signal.empty and rsi_signal.iloc[-1] > 0:
            signal_indicators.append('RSI')
        if not obv_signal.empty and obv_signal.iloc[-1] > 0:
            signal_indicators.append('OBV')
        if not atr_signal.empty and atr_signal.iloc[-1] > 0:
            signal_indicators.append('ATR')
        if not bb_signal.empty and bb_signal.iloc[-1] > 0:
            signal_indicators.append('BB')
        
        # 최대 가능한 신호강도 계산 (정규화용) - EMA 제외
        max_possible_score = (
            self.params.get('rsi_weight', 1.0) +
            self.params.get('obv_weight', 1.0) +
            self.params.get('atr_weight', 1.0) +
            self.params.get('bb_weight', 1.0)
        )
        
        # 정규화값 계산
        normalized_score = min(signal_strength / max_possible_score, 1.0) if max_possible_score > 0 else 0
        
        # 신호 가중치 계산 (거래일 단위 카운트 사용)
        # 공식: sell_weight = coefficient × base^(sell_count - 1)
        sell_weight_base = self.params.get('sell_weight_base', 1.05)
        sell_weight_coefficient = self.params.get('sell_weight_coefficient', 0.1)
        # self.sell_count: 신호가 발생한 거래일 수 (같은 날 여러 신호는 1회로 카운트)
        if self.sell_count == 0:
            sell_weight = sell_weight_coefficient  # 첫 번째 매도 전에는 계수값
        else:
            sell_weight = sell_weight_coefficient * (sell_weight_base ** (self.sell_count - 1))
        
        # 가격 가중치 계산 (수익률 가중치)
        price_weight_exponent = self.params.get('price_weight_exponent', 2.0)
        price_weight = max(1.0, (1.0 + current_return) ** price_weight_exponent)
        
        # 시간 가중치 계산
        time_weight_max = self.params.get('time_weight_max', 2.0)
        time_weight_midpoint = self.params.get('time_weight_midpoint', 365)
        time_weight_slope = self.params.get('time_weight_slope', 0.025)
        sigmoid = 1 / (1 + math.exp(-time_weight_slope * (days_held - time_weight_midpoint)))
        time_weight = 1 + (time_weight_max - 1) * sigmoid
        
        # 매도 비율 계산 (가상)
        position_manager = SellPositionManager(initial_position=100.0)
        max_sell_ratio = self.params.get('max_sell_ratio', 0.01)
        current_position_ratio = 1.0  # 100% 보유 가정

        # 전체 기준 매도비율 계산
        if signal_strength > 0:
            raw_sell_ratio = max_sell_ratio * normalized_score * sell_weight * current_position_ratio * price_weight * time_weight
            raw_sell_ratio = min(raw_sell_ratio, current_position_ratio)

            # 최소 매도비율 체크 (5% 미만이면 신호 없음)
            min_sell_ratio = current_position_ratio * 0.05
            total_sell_ratio = raw_sell_ratio if raw_sell_ratio >= min_sell_ratio else 0.0
        else:
            raw_sell_ratio = 0.0
            total_sell_ratio = 0.0

        # 보유 기준 매도비율 (현재 100% 보유이므로 전체 기준과 동일)
        hold_based_sell_ratio = total_sell_ratio / current_position_ratio if current_position_ratio > 0 else 0

        # 매도비율이 5% 이상일 때만 신호 발생으로 판단
        has_valid_signal = total_sell_ratio > 0

        return {
            'date': latest_date,
            'price': latest_price,
            'days_from_low': days_from_low,  # 저점 대비 보유일수
            'days_held': days_held,  # 매수일 기준 보유일수
            'signal_strength': signal_strength,
            'signal_indicators': signal_indicators,
            'normalized_score': normalized_score,  # 정규화값
            'sell_weight': sell_weight,  # 매도 가중치
            'price_weight': price_weight,  # 수익률 가중치
            'time_weight': time_weight,  # 시간 가중치
            'raw_sell_ratio': raw_sell_ratio,  # 임계값 적용 전 실제 계산값
            'total_sell_ratio': total_sell_ratio,  # 5% 임계값 적용 후 매도비율
            'hold_based_sell_ratio': hold_based_sell_ratio,  # 보유 기준 매도비율
            'max_possible_score': max_possible_score,  # 최대 가능 점수 (참고용)
            'has_signal': has_valid_signal  # 매도비율 5% 이상일 때만 True
        }


def test_detector():
    """신호 감지 테스트"""
    from data_fetcher import DataFetcher
    from config import SYMBOLS
    
    print("=" * 60)
    print("매도 신호 감지 테스트")
    print("=" * 60)
    
    for symbol_key, symbol_config in SYMBOLS.items():
        print(f"\n[{symbol_config['name']}]")
        
        # 데이터 수집
        fetcher = DataFetcher(
            symbol_config['ticker'],
            symbol_config['buy_date']
        )
        data_info = fetcher.get_latest_data()
        
        if data_info is None:
            print("데이터 수집 실패")
            continue
        
        # 신호 감지
        detector = SignalDetector(
            data_info['data'],
            symbol_config['params'],
            symbol_config['buy_date'],
            symbol_config['buy_price']
        )
        
        signal = detector.detect_signal()
        
        if signal:
            print(f"날짜: {signal['date'].strftime('%Y-%m-%d')}")
            print(f"현재가: ${signal['price']:.2f}")
            print(f"보유일수: {signal['days_held']}일 (저점 대비: {signal['days_from_low']}일)")
            print(f"신호 강도: {signal['signal_strength']:.4f} (최대: {signal['max_possible_score']:.0f})")
            print(f"신호 지표: {'+'.join(signal['signal_indicators']) if signal['signal_indicators'] else 'N/A'}")
            print(f"정규화값: {signal['normalized_score']:.4f}")
            print(f"매도가중치: {signal['sell_weight']:.4f}")
            print(f"수익률가중치: {signal['price_weight']:.4f}")
            print(f"시간가중치: {signal['time_weight']:.4f}")
            print(f"전체기준 매도비율: {signal['total_sell_ratio']*100:.2f}%")
            print(f"보유기준 매도비율: {signal['hold_based_sell_ratio']*100:.2f}%")
            print(f"⚠️ 매도 신호: {'있음' if signal['has_signal'] else '없음'}")


if __name__ == "__main__":
    test_detector()

