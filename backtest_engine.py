# =============================================================================
# 백테스트 엔진
# =============================================================================
import pandas as pd
import numpy as np
from sell_signal_generator import SellSignalGenerator
from sell_position_manager import SellPositionManager

class BacktestEngine:
    """백테스트 엔진 클래스"""
    
    def __init__(self, data, signal_generator):
        """
        Args:
            data: 기술적 지표가 포함된 데이터프레임
            signal_generator: SellSignalGenerator 인스턴스
        """
        self.data = data.copy()
        self.signal_generator = signal_generator
        # 최고점 가격 미리 계산
        self.peak_price = self.data['Close'].max()
    
    def run_backtest(self, params, include_sell_records=True, buy_avg_price=None):
        """
        백테스트 실행
        
        Args:
            params: 파라미터 딕셔너리
            include_sell_records: sell_records를 결과에 포함할지 여부 (메모리 절약용)
            buy_avg_price: 매수 평균단가 (None이면 첫 거래일 종가 사용)
        
        Returns:
            백테스트 결과 딕셔너리
        """
        # 매도 비중 관리자 초기화
        position_manager = SellPositionManager(initial_position=100.0)
        
        # 매수 평균단가 설정
        if buy_avg_price is None:
            # 매수 평균단가가 제공되지 않으면 첫 거래일 종가 사용
            initial_price = self.data['Close'].iloc[0]
        else:
            initial_price = buy_avg_price
        
        # 매도 기록
        sell_records = []
        
        # 신호 카운터 (초반 5번까지는 매도 진행하지 않음)
        signal_count = 0
        
        # 개별 지표별 신호 사전 계산 (한 번만 계산하여 중복 방지)
        rsi_signal = self.signal_generator.generate_rsi_signal(
            params.get('rsi_overbought', 70),
            params.get('rsi_period', 14),
            params.get('rsi_weight', 1.0)
        )
        obv_signal = self.signal_generator.generate_obv_signal(
            params.get('obv_period', 20),
            params.get('obv_weight', 1.0)
        )
        atr_signal = self.signal_generator.generate_atr_signal(
            params.get('atr_multiplier', 2.0),
            params.get('atr_weight', 1.0)
        )
        bb_signal = self.signal_generator.generate_bollinger_signal(
            params.get('bb_position_threshold', 100),
            params.get('bb_period', 20),
            params.get('bb_std_dev', 2.0),
            params.get('bb_weight', 1.0)
        )
        ema_signal = self.signal_generator.generate_ema_signal(
            params.get('ema_period', 60)
        )
        
        # 종합 기술점수 = 개별 신호의 합 (중복 계산 방지)
        combined_signal = rsi_signal + obv_signal + atr_signal + bb_signal + ema_signal
        if isinstance(combined_signal, np.ndarray):
            combined_signal = pd.Series(combined_signal, index=self.data.index)
        
        # 첫 거래일 저장 (보유 일수 계산용)
        first_date = self.data.index[0]
        
        # 각 거래일별로 순회
        for i, (date, row) in enumerate(self.data.iterrows()):
            # 보유 일수 계산
            days_held = (date - first_date).days
            
            # 종합 기술점수
            if isinstance(combined_signal, pd.Series):
                technical_score = combined_signal.iloc[i]
            elif isinstance(combined_signal, np.ndarray):
                technical_score = combined_signal[i]
            else:
                technical_score = combined_signal[i] if i < len(combined_signal) else 0
            
            # 개별 지표 신호값 가져오기
            rsi_val = rsi_signal.iloc[i] if isinstance(rsi_signal, pd.Series) else rsi_signal[i]
            obv_val = obv_signal.iloc[i] if isinstance(obv_signal, pd.Series) else obv_signal[i]
            atr_val = atr_signal.iloc[i] if isinstance(atr_signal, pd.Series) else atr_signal[i]
            bb_val = bb_signal.iloc[i] if isinstance(bb_signal, pd.Series) else bb_signal[i]
            
            # EMA 하향돌파 신호
            if isinstance(ema_signal, pd.Series):
                ema_sell_signal = ema_signal.iloc[i]
            elif isinstance(ema_signal, np.ndarray):
                ema_sell_signal = ema_signal[i]
            else:
                ema_sell_signal = ema_signal[i] if i < len(ema_signal) else 0
            
            # 신호 발생 여부 확인
            has_signal = False
            
            # 현재 수익률 계산
            current_return = (row['Close'] - initial_price) / initial_price
            
            # EMA 하향돌파 매도 - 비활성화 (매도비율 0%)
            # EMA 신호는 카운팅만 하고 실제 매도는 진행하지 않음
            if ema_sell_signal > 0 and current_return >= 1.0:  # 100% 이상
                has_signal = True
                signal_count += 1
                # EMA 매도는 비활성화 - 매도 진행하지 않음
                # (기술적 지표 기반 매도만 진행)
            
            # 종합 기술점수 기반 매도 (평균단가의 2배 이상, 즉 수익률 100% 이상에서만)
            if technical_score > 0 and current_return >= 1.0:  # 100% 이상
                if not has_signal:  # EMA 신호와 중복되지 않을 때만 카운트
                    signal_count += 1
                # 초반 5번까지는 매도 진행하지 않고 6번째 시그널부터 매도 진행
                if signal_count > 5:
                    # 수익률 가중치 계산 (지수 형태: (1.0 + return_ratio) ^ exponent)
                    price_weight_exponent = params.get('price_weight_exponent', 2.0)
                    price_weight = max(1.0, (1.0 + current_return) ** price_weight_exponent)
                    
                    # 시간 가중치 계산 (시그모이드 기반)
                    import math
                    time_weight_max = params.get('time_weight_max', 2.0)
                    time_weight_midpoint = params.get('time_weight_midpoint', 365)
                    time_weight_slope = params.get('time_weight_slope', 0.025)
                    sigmoid = 1 / (1 + math.exp(-time_weight_slope * (days_held - time_weight_midpoint)))
                    time_weight = 1 + (time_weight_max - 1) * sigmoid
                    
                    sell_ratio = position_manager.calculate_sell_ratio(technical_score, params, current_price=row['Close'], initial_price=initial_price, days_held=days_held)
                    if sell_ratio > 0:
                        actual_ratio = position_manager.execute_sell(
                            date=date,
                            price=row['Close'],
                            sell_ratio=sell_ratio,
                            signal_strength=technical_score,
                            signal_type='technical'
                        )
                        if actual_ratio > 0:
                            # 최고점 대비 매도가격 비율 계산
                            peak_price_ratio = (row['Close'] / self.peak_price) * 100.0
                            
                            # 신호를 발생시킨 지표 목록 생성
                            signal_indicators = []
                            if rsi_val > 0:
                                signal_indicators.append('RSI')
                            if obv_val > 0:
                                signal_indicators.append('OBV')
                            if atr_val > 0:
                                signal_indicators.append('ATR')
                            if bb_val > 0:
                                signal_indicators.append('BB')
                            signal_indicators_str = '+'.join(signal_indicators) if signal_indicators else 'N/A'
                            
                            sell_records.append({
                                'date': date,
                                'price': row['Close'],
                                'ratio': actual_ratio,
                                'amount': actual_ratio * 100.0,
                                'signal_strength': technical_score,
                                'signal_type': 'technical',
                                'signal_indicators': signal_indicators_str,  # 신호 발생 지표 추가
                                'sell_count': position_manager.sell_count,
                                'price_weight': price_weight,
                                'time_weight': time_weight,  # 시간 가중치 추가
                                'days_held': days_held,  # 보유 일수 추가
                                'peak_price_ratio': peak_price_ratio  # 최고점 대비 매도가격 비율 (%)
                            })
                        else:
                            # 매도가 실행되지 않았지만 매도가중치 카운팅
                            position_manager.increment_sell_count()
                    else:
                        # 매도 비율이 1% 미만이어서 매도하지 않았지만 매도가중치 카운팅
                        position_manager.increment_sell_count()
        
        # 성과 계산
        performance = self._calculate_performance(
            sell_records=sell_records,
            initial_price=initial_price,
            final_price=self.data['Close'].iloc[-1],
            final_position=position_manager.current_position
        )
        
        result = {
            'performance': performance,
            'position_manager_status': position_manager.get_status()
        }
        
        # 메모리 절약을 위해 선택적으로 sell_records 포함
        if include_sell_records:
            result['sell_records'] = sell_records
        
        return result
    
    def _calculate_performance(self, sell_records, initial_price, final_price, final_position):
        """성과 지표 계산"""
        if not sell_records:
            return {
                'total_return': 0.0,
                'num_trades': 0,
                'avg_signal_strength': 0.0,
                'sell_efficiency': 0.0,
                'avg_price': initial_price,
                'final_position': final_position,
                'peak_ratio_weighted_sum': 0.0  # 최고점 대비 매도비율의 합
            }
        
        df_sells = pd.DataFrame(sell_records)
        
        # 매도된 비중
        total_sold = df_sells['amount'].sum()
        
        # 가중 평균 매도 가격
        weighted_avg_price = (df_sells['price'] * df_sells['amount']).sum() / df_sells['amount'].sum()
        
        # 수익률 계산 (매도된 부분)
        sold_return = ((weighted_avg_price - initial_price) / initial_price) * (total_sold / 100.0)
        
        # 남은 부분의 수익률
        remaining_return = ((final_price - initial_price) / initial_price) * (final_position / 100.0)
        
        # 총 수익률
        total_return = sold_return + remaining_return
        
        # 평균 신호 강도
        avg_signal_strength = df_sells['signal_strength'].mean()
        
        # 매도 효율성 (전체 중 매도된 비율)
        sell_efficiency = total_sold / 100.0
        
        # 최고점 대비 매도비율의 합 계산
        # = Σ(최고점대비매도가격비율 × 매도비중)
        # 예: 82.20% × 7.08% = 5.82, 모든 매도에 대해 합산
        if 'peak_price_ratio' in df_sells.columns:
            # peak_price_ratio는 %, amount도 %이므로 100으로 나눠서 정규화
            peak_ratio_weighted_sum = (df_sells['peak_price_ratio'] * df_sells['amount'] / 100.0).sum()
        else:
            peak_ratio_weighted_sum = 0.0
        
        return {
            'total_return': total_return,
            'num_trades': len(sell_records),
            'avg_signal_strength': avg_signal_strength,
            'sell_efficiency': sell_efficiency,
            'avg_price': weighted_avg_price,
            'final_position': final_position,
            'total_sold': total_sold,
            'sold_return': sold_return,
            'remaining_return': remaining_return,
            'peak_ratio_weighted_sum': peak_ratio_weighted_sum  # 최고점 대비 매도비율의 합
        }

