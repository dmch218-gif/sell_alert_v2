# =============================================================================
# 매도 비중 관리 시스템
# =============================================================================
import pandas as pd
import numpy as np
import math

class SellPositionManager:
    """매도 비중 관리 클래스"""
    
    def __init__(self, initial_position=100.0):
        """
        Args:
            initial_position: 초기 보유 비중 (기본 100%)
        """
        self.initial_position = initial_position
        self.current_position = initial_position
        self.sell_count = 0  # 매도 횟수
        self.sell_history = []  # 매도 기록
    
    def calculate_sell_weight(self, sell_count, sell_weight_base=1.05, sell_weight_coefficient=0.1):
        """
        매도 횟수에 따른 지수형태 가중치 계산 (sell_weight_coefficient * sell_weight_base^(k-1))
        
        Args:
            sell_count: 현재까지의 매도 횟수 (k)
            sell_weight_base: 가중치 베이스 (기본 1.05)
            sell_weight_coefficient: 매도가중치 계수 (기본 0.1)
        
        Returns:
            가중치 값 (sell_weight_coefficient * sell_weight_base^(k-1))
        """
        if sell_count == 0:
            return sell_weight_coefficient  # 첫 번째 매도 전에는 계수값
        weight = sell_weight_coefficient * (sell_weight_base ** (sell_count - 1))
        return weight
    
    def calculate_time_weight(self, days_held, time_weight_max=2.0, time_weight_midpoint=365, time_weight_slope=0.025):
        """
        시그모이드 기반 시간 가중치 계산
        - 1년(365일)까지 완만하게 증가
        - 1년 이후 급격히 증가
        - 1년 반(~548일)에서 최대치에 도달
        
        Args:
            days_held: 보유 일수
            time_weight_max: 최대 시간 가중치 (기본 2.0 = 최대 2배)
            time_weight_midpoint: 중간점 - 50% 도달 시점 (기본 365일 = 1년)
            time_weight_slope: 기울기 계수 (기본 0.025, 1년 반에 ~99% 도달)
        
        Returns:
            시간 가중치 (1.0 ~ time_weight_max)
        """
        # 시그모이드 함수: 0~1 범위
        sigmoid = 1 / (1 + math.exp(-time_weight_slope * (days_held - time_weight_midpoint)))
        
        # 1 ~ time_weight_max 범위로 스케일링
        time_weight = 1 + (time_weight_max - 1) * sigmoid
        
        return time_weight
    
    def calculate_sell_ratio(self, technical_score, params, current_price=None, initial_price=None, days_held=0):
        """
        종합 기술점수와 매도 횟수 가중치를 고려한 매도 비율 계산
        
        Args:
            technical_score: 종합 기술점수
            params: 파라미터 딕셔너리
                - max_sell_ratio: 최대 매도 비율 (기본 0.02 = 2%)
                - sell_weight_base: 매도 가중치 베이스 (기본 1.05)
                - price_weight_enabled: 가격 가중치 사용 여부 (기본 True)
                - price_weight_base: 가격 가중치 베이스 (기본 1.0)
                - time_weight_max: 최대 시간 가중치 (기본 2.0)
                - time_weight_midpoint: 시간 가중치 중간점 (기본 365일)
                - time_weight_slope: 시간 가중치 기울기 (기본 0.025)
            current_price: 현재 가격 (가격 가중치 계산용)
            initial_price: 매수 평균단가 (가격 가중치 계산용)
            days_held: 보유 일수 (시간 가중치 계산용)
        
        Returns:
            매도 비율 (0-1 사이, 남은 금액 기준)
        """
        # 신호가 없으면 매도하지 않음
        if technical_score <= 0:
            return 0.0
        
        # 최대 가능한 신호강도 계산 (모든 지표가 동시에 발생할 때)
        max_possible_score = (
            params.get('rsi_weight', 1.0) +
            params.get('obv_weight', 1.0) +
            params.get('atr_weight', 1.0) +
            params.get('bb_weight', 1.0) +
            1.0  # EMA는 고정 1.0
        )
        
        # 기술점수를 0-1 범위로 정규화 (0 ~ 최대값 기준)
        normalized_score = min(technical_score / max_possible_score, 1.0)
        
        # 매도 횟수에 따른 지수형태 가중치 계산
        sell_weight_base = params.get('sell_weight_base', 1.05)
        sell_weight_coefficient = params.get('sell_weight_coefficient', 0.1)
        sell_weight = self.calculate_sell_weight(self.sell_count, sell_weight_base, sell_weight_coefficient)
        
        # 최대 매도 비율 (1% 고정)
        max_sell_ratio = params.get('max_sell_ratio', 0.01)
        
        # 현재 보유 비중을 백분율로 변환 (예: 80% -> 0.80)
        current_position_ratio = self.current_position / 100.0
        
        # 가격 가중치 계산 (현재 가격이 높을수록 더 많이 매도)
        price_weight = 1.0
        if current_price is not None and initial_price is not None and initial_price > 0:
            # 수익률 계산 (100% = 2배)
            return_ratio = (current_price - initial_price) / initial_price
            # 수익률 가중치 지수 (그리드 서치 파라미터)
            price_weight_exponent = params.get('price_weight_exponent', 2.0)
            # 수익률 가중치 계산: (1.0 + return_ratio) ^ exponent
            price_weight = max(1.0, (1.0 + return_ratio) ** price_weight_exponent)
        
        # 시간 가중치 계산 (시그모이드 기반)
        time_weight_max = params.get('time_weight_max', 2.0)
        time_weight_midpoint = params.get('time_weight_midpoint', 365)
        time_weight_slope = params.get('time_weight_slope', 0.025)
        time_weight = self.calculate_time_weight(days_held, time_weight_max, time_weight_midpoint, time_weight_slope)
        
        # 최종 매도 비율 = 최대매도비율 * 정규화된 기술점수 * 매도 가중치 * 현재 보유 비중 * 가격 가중치 * 시간 가중치
        sell_ratio = max_sell_ratio * normalized_score * sell_weight * current_position_ratio * price_weight * time_weight
        
        # 현재 보유 비중을 초과하지 않도록 제한
        sell_ratio = min(sell_ratio, current_position_ratio)
        
        # 매도 비율이 현재 보유 비중의 5% 미만이면 매도하지 않음
        min_sell_ratio = current_position_ratio * 0.05
        if sell_ratio < min_sell_ratio:
            return 0.0
        
        return sell_ratio
    
    def execute_ema_sell(self, params, current_price=None, initial_price=None, days_held=0):
        """
        EMA 하향돌파 시 매도 실행 (남은 금액 기준)
        
        Args:
            params: 파라미터 딕셔너리
                - max_sell_ratio: 최대 매도 비율 (기본 0.02 = 2%)
                - sell_weight_base: 매도 가중치 베이스 (기본 1.05)
                - time_weight_max: 최대 시간 가중치 (기본 2.0)
                - time_weight_midpoint: 시간 가중치 중간점 (기본 365일)
                - time_weight_slope: 시간 가중치 기울기 (기본 0.025)
            current_price: 현재 가격 (가격 가중치 계산용)
            initial_price: 매수 평균단가 (가격 가중치 계산용)
            days_held: 보유 일수 (시간 가중치 계산용)
        
        Returns:
            매도 비율 (0-1 사이, 남은 금액 기준)
        """
        # 최대 매도 비율 (1% 고정)
        max_sell_ratio = params.get('max_sell_ratio', 0.01)
        
        # 매도 가중치 계산
        sell_weight_base = params.get('sell_weight_base', 1.05)
        sell_weight = self.calculate_sell_weight(self.sell_count, sell_weight_base)
        
        # EMA 신호는 정규화값 0.1로 처리
        normalized_score = 0.1
        
        # 현재 보유 비중을 백분율로 변환 (예: 80% -> 0.80)
        current_position_ratio = self.current_position / 100.0
        
        # 가격 가중치 계산 (현재 가격이 높을수록 더 많이 매도)
        price_weight = 1.0
        if current_price is not None and initial_price is not None and initial_price > 0:
            # 수익률 계산 (100% = 2배)
            return_ratio = (current_price - initial_price) / initial_price
            # 수익률 가중치 지수 (그리드 서치 파라미터)
            price_weight_exponent = params.get('price_weight_exponent', 2.0)
            # 수익률 가중치 계산: (1.0 + return_ratio) ^ exponent
            price_weight = max(1.0, (1.0 + return_ratio) ** price_weight_exponent)
        
        # 시간 가중치 계산 (시그모이드 기반)
        time_weight_max = params.get('time_weight_max', 2.0)
        time_weight_midpoint = params.get('time_weight_midpoint', 365)
        time_weight_slope = params.get('time_weight_slope', 0.025)
        time_weight = self.calculate_time_weight(days_held, time_weight_max, time_weight_midpoint, time_weight_slope)
        
        # 최종 매도 비율 = 최대매도비율 * 정규화된 기술점수 * 매도 가중치 * 현재 보유 비중 * 가격 가중치 * 시간 가중치
        sell_ratio = max_sell_ratio * normalized_score * sell_weight * current_position_ratio * price_weight * time_weight
        
        # 현재 보유 비중을 초과하지 않도록 제한
        sell_ratio = min(sell_ratio, current_position_ratio)
        
        # 매도 비율이 현재 보유 비중의 5% 미만이면 매도하지 않음
        min_sell_ratio = current_position_ratio * 0.05
        if sell_ratio < min_sell_ratio:
            return 0.0
        
        return sell_ratio
    
    def execute_sell(self, date, price, sell_ratio, signal_strength, signal_type='technical'):
        """
        매도 실행
        
        Args:
            date: 매도 일자
            price: 매도 가격
            sell_ratio: 매도 비율 (0-1)
            signal_strength: 신호 강도
            signal_type: 신호 유형 ('technical' 또는 'ema')
        
        Returns:
            실제 매도된 비율
        """
        if sell_ratio <= 0:
            return 0.0
        
        # 실제 매도 비율 계산
        actual_sell_ratio = min(sell_ratio, self.current_position / 100.0)
        
        if actual_sell_ratio > 0:
            # 매도 실행
            sell_amount = actual_sell_ratio * 100.0  # 비율을 퍼센트로 변환
            
            # 매도 후 남은 보유 비중 계산
            remaining_after_sell = self.current_position - sell_amount
            
            # 남은 보유 비중이 1% 이하인 경우, 남은 비중 전체 매도
            if remaining_after_sell <= 1.0 and remaining_after_sell > 0:
                sell_amount = self.current_position  # 남은 비중 전체 매도
                actual_sell_ratio = self.current_position / 100.0
                remaining_after_sell = 0.0
            
            # 기록 저장
            sell_record = {
                'date': date,
                'price': price,
                'ratio': actual_sell_ratio,
                'amount': sell_amount,
                'signal_strength': signal_strength,
                'signal_type': signal_type,
                'sell_count': self.sell_count + 1,
                'remaining_position': remaining_after_sell
            }
            
            self.sell_history.append(sell_record)
            
            # 상태 업데이트
            self.current_position = remaining_after_sell
            self.sell_count += 1
            
            return actual_sell_ratio
        
        return 0.0
    
    def increment_sell_count(self):
        """
        매도가 실행되지 않더라도 매도 카운트 증가 (매도가중치 계산용)
        """
        self.sell_count += 1
    
    def get_sell_history(self):
        """매도 기록 반환"""
        return self.sell_history.copy()
    
    def reset(self):
        """상태 초기화"""
        self.current_position = self.initial_position
        self.sell_count = 0
        self.sell_history = []
    
    def get_status(self):
        """현재 상태 반환"""
        return {
            'current_position': self.current_position,
            'sell_count': self.sell_count,
            'total_sold': self.initial_position - self.current_position
        }

