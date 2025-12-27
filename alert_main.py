# =============================================================================
# 매도 신호 알림 시스템 - 메인 실행 스크립트 (실시간 모니터링 버전)
# =============================================================================
import sys
import time
import schedule
from datetime import datetime, timedelta
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SYMBOLS,
    SIGNAL_THRESHOLD, ALERT_COOLDOWN_HOURS,
    CHECK_INTERVAL_MINUTES, OPERATING_HOURS, DAILY_SUMMARY_TIME,
    US_MARKET_HOURS
)
from data_fetcher import DataFetcher
from signal_detector import SignalDetector
from notifier import TelegramNotifier
from signal_history_manager import SignalHistoryManager


class RealtimeSellSignalSystem:
    """실시간 매도 신호 알림 시스템"""
    
    def __init__(self):
        """시스템 초기화"""
        self.notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.signal_history_manager = SignalHistoryManager()  # 거래일 단위 신호 이력 관리
        self.last_check_time = None
        self.daily_signal_history = {}  # 금일 신호 이력 {symbol: [signals]}
        self.last_summary_date = None   # 마지막 요약 전송 날짜
        self.cached_data = {}           # 종목별 캐시된 일봉 데이터
        self.cache_time = {}            # 캐시 시간
        self.last_signal_time = {}      # 종목별 마지막 신호 시간 (중복 알림 방지)
        self.cached_sell_count = {}     # 종목별 캐시된 과거 신호 횟수
        self.sell_count_cache_time = {} # 과거 신호 캐시 시간
        
    def is_market_hours(self) -> bool:
        """
        미국 장 시간 체크 (한국 시간 기준)
        
        Returns:
            장 운영 시간 여부
        """
        current_hour = datetime.now().hour
        
        # 운영 시간이 자정을 넘는 경우 (예: 22시 ~ 08시)
        if OPERATING_HOURS['start'] > OPERATING_HOURS['end']:
            return current_hour >= OPERATING_HOURS['start'] or current_hour < OPERATING_HOURS['end']
        else:
            return OPERATING_HOURS['start'] <= current_hour < OPERATING_HOURS['end']
    
    def is_summary_time(self) -> bool:
        """일일 요약 전송 시간인지 확인"""
        now = datetime.now()
        
        # 이미 오늘 요약을 보냈는지 확인
        if self.last_summary_date == now.date():
            return False
        
        # 요약 시간 체크 (정확한 시간 또는 1분 이내)
        if now.hour == DAILY_SUMMARY_TIME['hour'] and now.minute == DAILY_SUMMARY_TIME['minute']:
            return True
        
        return False
    
    def get_cached_data(self, symbol_key: str, symbol_config: dict, max_cache_minutes: int = 30) -> dict:
        """
        캐시된 일봉 데이터 가져오기 (API 호출 최소화)
        
        Args:
            symbol_key: 종목 코드
            symbol_config: 종목 설정
            max_cache_minutes: 캐시 유효 시간 (분)
        
        Returns:
            데이터 정보 딕셔너리
        """
        now = datetime.now()
        
        # 캐시 유효성 확인
        if symbol_key in self.cached_data and symbol_key in self.cache_time:
            elapsed = (now - self.cache_time[symbol_key]).total_seconds() / 60
            if elapsed < max_cache_minutes:
                return self.cached_data[symbol_key]
        
        # 새 데이터 가져오기
        fetcher = DataFetcher(
            symbol_config['ticker'],
            symbol_config['buy_date']
        )
        data_info = fetcher.get_latest_data()
        
        if data_info is not None:
            self.cached_data[symbol_key] = data_info
            self.cache_time[symbol_key] = now
        
        return data_info
    
    def calculate_historical_signals(self, symbol_key: str, symbol_config: dict, data) -> dict:
        """
        4/11부터 현재까지의 과거 종가 기준 신호를 계산
        
        파라미터가 변경되어도 항상 정확한 값을 반환
        
        Args:
            symbol_key: 종목 코드
            symbol_config: 종목 설정
            data: 전체 일봉 데이터
        
        Returns:
            {
                'sell_count': 신호 발생 거래일 수,
                'sell_history': 매도 이력 테이블 리스트
            }
        """
        import pandas as pd
        import math
        from sell_signal_generator import SellSignalGenerator
        
        params = symbol_config['params']
        buy_date = pd.to_datetime(symbol_config['buy_date'])
        buy_price = symbol_config.get('buy_price')
        
        # 매수가 계산 (None이면 매수일 종가의 2배)
        buy_date_data = data[data['Date'] >= buy_date]
        if buy_date_data.empty:
            return {'sell_count': 0, 'sell_history': []}
        
        if buy_price is None:
            buy_price = buy_date_data.iloc[0]['Close'] * 2.0
        
        # 신호 생성기 초기화
        signal_gen = SellSignalGenerator(data)
        
        # 각 지표별 신호 계산
        rsi_signal = signal_gen.generate_rsi_signal(
            params.get('rsi_overbought', 70),
            params.get('rsi_period', 14),
            params.get('rsi_weight', 1.0)
        )
        
        bb_signal = signal_gen.generate_bollinger_signal(
            params.get('bb_position_threshold', 100),
            params.get('bb_period', 20),
            params.get('bb_std_dev', 2.0),
            params.get('bb_weight', 1.0)
        )
        
        obv_signal = signal_gen.generate_obv_signal(
            params.get('obv_period', 20),
            params.get('obv_weight', 1.0)
        )
        
        atr_signal = signal_gen.generate_atr_signal(
            params.get('atr_multiplier', 2.0),
            params.get('atr_weight', 1.0)
        )
        
        # 통합 신호
        combined = rsi_signal + bb_signal + obv_signal + atr_signal
        
        # 최대 가능 점수 (EMA 제외)
        max_possible_score = (
            params.get('rsi_weight', 1.0) +
            params.get('obv_weight', 1.0) +
            params.get('atr_weight', 1.0) +
            params.get('bb_weight', 1.0)
        )
        
        # 파라미터
        max_sell_ratio = params.get('max_sell_ratio', 0.01)
        sell_weight_base = params.get('sell_weight_base', 1.05)
        sell_weight_coefficient = params.get('sell_weight_coefficient', 0.1)
        price_weight_exponent = params.get('price_weight_exponent', 2.0)
        time_weight_max = params.get('time_weight_max', 2.0)
        time_weight_midpoint = params.get('time_weight_midpoint', 365)
        time_weight_slope = params.get('time_weight_slope', 0.025)
        
        sell_history = []
        sell_count = 0
        cumulative_ratio = 0.0
        current_position = 1.0  # 100%
        
        for i in range(len(data)):
            row = data.iloc[i]
            date = row['Date']
            price = row['Close']
            
            # 매수일 이전은 스킵
            if date < buy_date:
                continue
            
            signal_strength = combined.iloc[i]
            if signal_strength <= 0:
                continue
            
            # 수익률 계산
            current_return = (price - buy_price) / buy_price
            
            # 수익률 100% 미만이면 스킵
            if current_return < 1.0:
                continue
            
            # 보유일수 계산
            days_held = (date - buy_date).days
            
            # 정규화값
            normalized_score = min(signal_strength / max_possible_score, 1.0) if max_possible_score > 0 else 0
            
            # 신호가중치 계산: coefficient × base^(sell_count - 1)
            if sell_count == 0:
                sell_weight = sell_weight_coefficient
            else:
                sell_weight = sell_weight_coefficient * (sell_weight_base ** (sell_count - 1))
            
            # 수익률 가중치
            price_weight = max(1.0, (1.0 + current_return) ** price_weight_exponent)
            
            # 시간 가중치
            sigmoid = 1 / (1 + math.exp(-time_weight_slope * (days_held - time_weight_midpoint)))
            time_weight = 1 + (time_weight_max - 1) * sigmoid
            
            # 매도비율 계산
            sell_ratio = max_sell_ratio * normalized_score * sell_weight * current_position * price_weight * time_weight
            sell_ratio = min(sell_ratio, current_position)
            
            # 최소 매도비율 체크 (5% 미만이면 0)
            min_sell_ratio = current_position * 0.05
            if sell_ratio < min_sell_ratio:
                continue
            
            # 유효 신호 발생!
            sell_count += 1
            cumulative_ratio += sell_ratio
            current_position = max(0, 1.0 - cumulative_ratio)
            
            sell_history.append({
                'no': sell_count,
                'date': date.strftime('%Y-%m-%d'),
                'price': price,
                'sell_ratio': sell_ratio,
                'cumulative_ratio': cumulative_ratio,
                'remaining_ratio': current_position
            })
        
        return {
            'sell_count': sell_count,
            'sell_history': sell_history
        }
    
    def check_realtime_signals(self):
        """
        실시간 신호 체크 (매분 실행)
        - 신호가 있을 때만 알림 전송
        """
        now = datetime.now()
        
        for symbol_key, symbol_config in SYMBOLS.items():
            try:
                # 캐시된 일봉 데이터 가져오기
                data_info = self.get_cached_data(symbol_key, symbol_config)
                
                if data_info is None:
                    continue
                
                # 실시간 가격 가져오기
                fetcher = DataFetcher(symbol_config['ticker'], symbol_config['buy_date'])
                realtime = fetcher.get_realtime_price()
                
                if realtime is None:
                    continue
                
                current_price = realtime['price']
                
                # 과거 신호 횟수 가져오기 (캐시 사용, 1시간마다 갱신)
                cache_valid = False
                if symbol_key in self.cached_sell_count and symbol_key in self.sell_count_cache_time:
                    elapsed = (now - self.sell_count_cache_time[symbol_key]).total_seconds() / 3600
                    cache_valid = elapsed < 1.0  # 1시간 캐시
                
                if not cache_valid:
                    # 과거 데이터로 신호 횟수 계산
                    historical = self.calculate_historical_signals(symbol_key, symbol_config, data_info['data'])
                    self.cached_sell_count[symbol_key] = historical['sell_count']
                    self.sell_count_cache_time[symbol_key] = now
                
                sell_count = self.cached_sell_count.get(symbol_key, 0)
                
                # 신호 감지 (일봉 데이터 기반 + 현재가 적용 + 과거 신호 횟수 반영)
                detector = SignalDetector(
                    data_info['data'],
                    symbol_config['params'],
                    symbol_config['buy_date'],
                    symbol_config['buy_price'],
                    sell_count=sell_count  # 거래일 단위 신호 카운트 전달
                )
                
                # 현재가를 기준으로 신호 감지
                signal = detector.detect_signal_with_price(current_price)
                
                if signal is None:
                    continue
                
                # 신호 발생 시 (매도비율 5% 이상)
                if signal['has_signal']:
                    # 중복 알림 방지 (같은 종목에 대해 30분 내 재알림 방지)
                    if symbol_key in self.last_signal_time:
                        elapsed = (now - self.last_signal_time[symbol_key]).total_seconds() / 60
                        if elapsed < 30:  # 30분 내 중복 방지
                            continue
                    
                    # 실시간 신호는 참고용이므로 별도 기록하지 않음
                    # (과거 이력은 매번 calculate_historical_signals로 계산)
                    
                    # 금일 신호 이력 기록 (참고용 알림 전송을 위해)
                    if symbol_key not in self.daily_signal_history:
                        self.daily_signal_history[symbol_key] = []
                    
                    # 금일 누적 매도비율 계산 (이번 매도 포함)
                    prev_cumulative = sum(sig['ratio'] for sig in self.daily_signal_history[symbol_key])
                    cumulative_sell_ratio = prev_cumulative + signal['total_sell_ratio']
                    remaining_after_sell = max(0, 1.0 - cumulative_sell_ratio)
                    
                    # 상세 정보 저장
                    self.daily_signal_history[symbol_key].append({
                        'time': now.strftime('%H:%M'),
                        'price': current_price,
                        'total_sell_ratio': signal['total_sell_ratio'],
                        'hold_based_sell_ratio': signal['hold_based_sell_ratio'],
                        'signal_indicators': signal.get('signal_indicators', []),
                        'signal_strength': signal['signal_strength'],
                        'normalized_score': signal['normalized_score'],
                        'sell_weight': signal['sell_weight'],
                        'price_weight': signal['price_weight'],
                        'time_weight': signal['time_weight'],
                        'days_held': signal['days_held'],
                        'remaining_position': remaining_after_sell,
                        'ratio': signal['total_sell_ratio']  # 호환성 유지
                    })
                    remaining_position = max(0, 100 - cumulative_sell_ratio * 100)
                    
                    # 전체 매도 이력 (과거 데이터 기반 계산 결과 사용)
                    historical = self.calculate_historical_signals(symbol_key, symbol_config, data_info['data'])
                    sell_history = historical.get('sell_history', [])
                    
                    print(f"\n🚨 [{symbol_config['name']}] 매도 신호 발생!")
                    print(f"   현재가: ${current_price:,.2f}")
                    print(f"   신호 강도: {signal['signal_strength']:.4f}")
                    print(f"   신호 가중치: {signal['sell_weight']:.4f} (거래일 {sell_count}회 기준)")
                    print(f"   매도비율: {signal['total_sell_ratio']*100:.2f}%")
                    print(f"   금일 누적: {cumulative_sell_ratio*100:.2f}% (잔여: {remaining_position:.2f}%)")
                    
                    # 알림 전송 (누적 매도비율 + 전체 매도 이력 포함)
                    self.notifier.send_realtime_signal_alert(
                        symbol_key,
                        symbol_config['name'],
                        signal,
                        current_price,
                        cumulative_sell_ratio,
                        sell_history
                    )
                    
                    self.last_signal_time[symbol_key] = now
                    
            except Exception as e:
                print(f"❌ {symbol_key} 체크 중 오류: {e}")
        
        self.last_check_time = now
    
    def send_daily_summary(self):
        """일일 요약 전송 (장 마감 후)"""
        print("\n" + "=" * 60)
        print(f"📊 일일 요약 생성 중... {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        summaries = []
        all_sell_history = {}
        
        for symbol_key, symbol_config in SYMBOLS.items():
            try:
                # 최신 일봉 데이터 가져오기 (캐시 무시)
                fetcher = DataFetcher(
                    symbol_config['ticker'],
                    symbol_config['buy_date']
                )
                data_info = fetcher.get_latest_data()
                
                if data_info is None:
                    print(f"❌ {symbol_key}: 데이터 수집 실패")
                    continue
                
                # 4/11부터 현재까지의 과거 신호 계산 (매번 현재 파라미터 기준으로 계산)
                historical = self.calculate_historical_signals(symbol_key, symbol_config, data_info['data'])
                sell_count = historical['sell_count']
                sell_history = historical['sell_history']
                
                print(f"📊 {symbol_key}: 과거 신호 {sell_count}회 발생")
                
                # 과거 이력 저장
                if sell_history:
                    all_sell_history[symbol_key] = sell_history
                
                # 오늘 신호 감지 (sell_count 반영)
                detector = SignalDetector(
                    data_info['data'],
                    symbol_config['params'],
                    symbol_config['buy_date'],
                    symbol_config['buy_price'],
                    sell_count=sell_count  # 과거 신호 횟수 반영
                )
                
                signal = detector.detect_signal()
                
                if signal is None:
                    continue
                
                # 전일 종가 대비 변화율 계산
                realtime = fetcher.get_realtime_price()
                change_percent = realtime['change_percent'] if realtime else 0
                
                summaries.append({
                    'symbol': symbol_key,
                    'name': symbol_config['name'],
                    'price': signal['price'],
                    'days_held': signal['days_held'],
                    'days_from_low': signal['days_from_low'],
                    'signal_strength': signal['signal_strength'],
                    'signal_indicators': signal.get('signal_indicators', []),
                    'total_sell_ratio': signal['total_sell_ratio'],
                    'has_signal': signal['has_signal'],
                    'change_percent': change_percent,
                    'sell_count': sell_count  # 과거 신호 횟수 추가
                })
                
                status = "🔴 신호 있음" if signal['has_signal'] else "🟢 신호 없음"
                print(f"✅ {symbol_config['name']}: {status} (신호가중치 기준: {sell_count}회)")
                
            except Exception as e:
                print(f"❌ {symbol_key} 요약 생성 오류: {e}")
        
        # 텔레그램으로 요약 전송 (전체 매도 이력 포함)
        if summaries:
            self.notifier.send_daily_summary(summaries, self.daily_signal_history, all_sell_history)
            print("📤 일일 요약 전송 완료")
        
        # 신호 이력 초기화 (다음 날을 위해)
        self.daily_signal_history = {}
        self.last_summary_date = datetime.now().date()
    
    def run_once(self):
        """한 번 실행 (테스트용)"""
        print("\n🔍 매도 신호 1회 체크 모드")
        
        for symbol_key, symbol_config in SYMBOLS.items():
            print(f"\n[{symbol_config['name']}]")
            
            try:
                fetcher = DataFetcher(
                    symbol_config['ticker'],
                    symbol_config['buy_date']
                )
                
                # 실시간 가격
                realtime = fetcher.get_realtime_price()
                if realtime:
                    print(f"  실시간 가격: ${realtime['price']:,.2f}")
                    print(f"  시장 상태: {realtime['market_state']}")
                    print(f"  전일 대비: {realtime['change_percent']:+.2f}%")
                
                # 일봉 기반 신호
                data_info = fetcher.get_latest_data()
                if data_info:
                    detector = SignalDetector(
                        data_info['data'],
                        symbol_config['params'],
                        symbol_config['buy_date'],
                        symbol_config['buy_price']
                    )
                    signal = detector.detect_signal()
                    
                    if signal:
                        print(f"  신호 강도: {signal['signal_strength']:.4f}")
                        print(f"  매도비율: {signal['total_sell_ratio']*100:.2f}%")
                        print(f"  신호 상태: {'🔴 있음' if signal['has_signal'] else '🟢 없음'}")
                
            except Exception as e:
                print(f"❌ 오류: {e}")
    
    def run_realtime(self):
        """실시간 모니터링 모드"""
        print("\n" + "=" * 60)
        print("🔄 실시간 모니터링 모드 시작")
        print("=" * 60)
        print(f"   체크 주기: {CHECK_INTERVAL_MINUTES}분")
        print(f"   운영 시간: {OPERATING_HOURS['start']}시 ~ {OPERATING_HOURS['end']}시")
        print(f"   일일 요약: {DAILY_SUMMARY_TIME['hour']:02d}:{DAILY_SUMMARY_TIME['minute']:02d}")
        print(f"   알림 방식: 신호 발생 시에만 알림")
        print("=" * 60)
        
        # 시작 알림
        if self.notifier.test_connection():
            self.notifier.send_message(
                f"🚀 <b>실시간 모니터링 시작</b>\n\n"
                f"📅 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"⏱️ 체크 주기: {CHECK_INTERVAL_MINUTES}분\n"
                f"📊 모니터링 종목: SOXL, USD\n\n"
                f"✅ 신호 발생 시에만 알림을 보내드립니다."
            )
        else:
            print("❌ 텔레그램 연결 실패. config.py 설정을 확인해주세요.")
            return
        
        # 스케줄 설정
        schedule.every(CHECK_INTERVAL_MINUTES).minutes.do(self._scheduled_check)
        schedule.every().day.at(f"{DAILY_SUMMARY_TIME['hour']:02d}:{DAILY_SUMMARY_TIME['minute']:02d}").do(self.send_daily_summary)
        
        print(f"\n⏰ 모니터링 시작 - 다음 체크: {CHECK_INTERVAL_MINUTES}분 후")
        print("   Ctrl+C로 종료할 수 있습니다.\n")
        
        # 즉시 1회 체크 (조용히)
        if self.is_market_hours():
            self.check_realtime_signals()
        
        while True:
            try:
                schedule.run_pending()
                time.sleep(30)  # 30초마다 스케줄 체크
            except KeyboardInterrupt:
                print("\n\n🛑 사용자에 의해 중단됨")
                self.notifier.send_message("🛑 모니터링 시스템이 중단되었습니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
                time.sleep(60)
    
    def _scheduled_check(self):
        """스케줄된 체크"""
        now = datetime.now()
        
        if self.is_market_hours():
            # 간단한 로그 (신호 없으면 최소한의 출력)
            print(f"[{now.strftime('%H:%M')}] 체크 중...", end=" ")
            
            signals_found = False
            for symbol_key, symbol_config in SYMBOLS.items():
                try:
                    data_info = self.get_cached_data(symbol_key, symbol_config)
                    if data_info is None:
                        continue
                    
                    fetcher = DataFetcher(symbol_config['ticker'], symbol_config['buy_date'])
                    realtime = fetcher.get_realtime_price()
                    
                    if realtime is None:
                        continue
                    
                    # 거래일 단위 신호 카운트 가져오기
                    sell_count = self.signal_history_manager.get_sell_count(symbol_key)
                    
                    detector = SignalDetector(
                        data_info['data'],
                        symbol_config['params'],
                        symbol_config['buy_date'],
                        symbol_config['buy_price'],
                        sell_count=sell_count
                    )
                    
                    signal = detector.detect_signal_with_price(realtime['price'])
                    
                    if signal and signal['has_signal']:
                        signals_found = True
                        
                        # 중복 방지 체크
                        if symbol_key in self.last_signal_time:
                            elapsed = (now - self.last_signal_time[symbol_key]).total_seconds() / 60
                            if elapsed < 30:
                                continue
                        
                        # 실시간 신호는 참고용이므로 전체 매도 이력에 기록하지 않음
                        # (종가 기준으로만 매도 이력 기록)
                        
                        # 금일 신호 이력 기록 (참고용)
                        if symbol_key not in self.daily_signal_history:
                            self.daily_signal_history[symbol_key] = []
                        
                        # 금일 누적 매도비율 계산
                        prev_cumulative = sum(sig['ratio'] for sig in self.daily_signal_history[symbol_key])
                        cumulative_sell_ratio = prev_cumulative + signal['total_sell_ratio']
                        remaining_after_sell = max(0, 1.0 - cumulative_sell_ratio)
                        
                        # 상세 정보 저장
                        self.daily_signal_history[symbol_key].append({
                            'time': now.strftime('%H:%M'),
                            'price': realtime['price'],
                            'total_sell_ratio': signal['total_sell_ratio'],
                            'hold_based_sell_ratio': signal['hold_based_sell_ratio'],
                            'signal_indicators': signal.get('signal_indicators', []),
                            'signal_strength': signal['signal_strength'],
                            'normalized_score': signal['normalized_score'],
                            'sell_weight': signal['sell_weight'],
                            'price_weight': signal['price_weight'],
                            'time_weight': signal['time_weight'],
                            'days_held': signal['days_held'],
                            'remaining_position': remaining_after_sell,
                            'ratio': signal['total_sell_ratio']
                        })
                        
                        # 전체 매도 이력 테이블 가져오기
                        sell_history = self.signal_history_manager.get_sell_history_table(symbol_key)
                        
                        print(f"\n🚨 [{symbol_config['name']}] 신호! (누적: {cumulative_sell_ratio*100:.2f}%, 거래일 {sell_count}회)")
                        
                        self.notifier.send_realtime_signal_alert(
                            symbol_key,
                            symbol_config['name'],
                            signal,
                            realtime['price'],
                            cumulative_sell_ratio,
                            sell_history
                        )
                        
                        self.last_signal_time[symbol_key] = now
                        
                except Exception as e:
                    print(f"오류({symbol_key}): {e}")
            
            if not signals_found:
                print("신호 없음")
        else:
            # 운영 시간 외
            if now.minute == 0:  # 매 시간마다만 로그
                print(f"[{now.strftime('%H:%M')}] 운영 시간 외 (대기 중)")


# =============================================================================
# 실시간 지표 계산 헬퍼 함수
# =============================================================================
def calculate_realtime_rsi(data, current_price: float, period: int = 14) -> float:
    """
    현재가를 반영한 실시간 RSI 계산
    
    Args:
        data: 기존 일봉 데이터
        current_price: 현재 실시간 가격
        period: RSI 기간
    
    Returns:
        실시간 RSI 값
    """
    # 마지막 Close를 현재가로 대체한 임시 시리즈 생성
    close_series = data['Close'].copy()
    close_series.iloc[-1] = current_price
    
    delta = close_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.iloc[-1]


def calculate_realtime_bb_position(data, current_price: float, period: int = 20, std_dev: float = 2.0) -> float:
    """
    현재가를 반영한 실시간 볼린저 밴드 위치 계산
    
    Args:
        data: 기존 일봉 데이터
        current_price: 현재 실시간 가격
        period: 볼린저 밴드 기간
        std_dev: 표준편차 배수
    
    Returns:
        실시간 BB Position 값 (0-100+)
    """
    # 볼린저 밴드는 과거 N일의 이동평균과 표준편차로 계산
    # 밴드 자체는 일봉 기준으로 유지하고, 현재가의 위치만 실시간으로 계산
    close_series = data['Close']
    
    # 밴드 계산 (과거 데이터 기준 - 당일 제외)
    sma = close_series.rolling(window=period).mean().iloc[-1]
    std = close_series.rolling(window=period).std().iloc[-1]
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    # 현재가의 밴드 내 위치 계산
    if upper_band != lower_band:
        bb_position = ((current_price - lower_band) / (upper_band - lower_band)) * 100
    else:
        bb_position = 50.0
    
    return bb_position


def generate_realtime_rsi_signal(data, current_price: float, rsi_overbought: float, rsi_period: int, weight: float) -> float:
    """
    실시간 RSI 신호 생성 (하향돌파 방식)
    """
    import numpy as np
    
    # 마지막 Close를 현재가로 대체한 임시 시리즈 생성
    close_series = data['Close'].copy()
    close_series.iloc[-1] = current_price
    
    delta = close_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # 전일 RSI (원본 데이터 기준)
    original_close = data['Close']
    original_delta = original_close.diff()
    original_gain = (original_delta.where(original_delta > 0, 0)).rolling(window=rsi_period).mean()
    original_loss = (-original_delta.where(original_delta < 0, 0)).rolling(window=rsi_period).mean()
    original_rs = original_gain / original_loss
    original_rsi = 100 - (100 / (1 + original_rs))
    
    rsi_prev = original_rsi.iloc[-1]  # 전일 종가 기준 RSI
    rsi_current = rsi.iloc[-1]  # 현재가 기준 RSI
    
    # 하향돌파: 전일 > overbought, 현재 <= overbought
    if rsi_prev > rsi_overbought and rsi_current <= rsi_overbought:
        return weight
    return 0.0


def generate_realtime_bb_signal(data, current_price: float, bb_threshold: float, bb_period: int, bb_std_dev: float, weight: float) -> float:
    """
    실시간 볼린저 밴드 신호 생성
    """
    bb_position = calculate_realtime_bb_position(data, current_price, bb_period, bb_std_dev)
    
    # BB_Position이 임계값 이상이면 신호 생성
    if bb_position >= bb_threshold:
        signal = min((bb_position - bb_threshold) / (100 - bb_threshold + 1e-10), 1.0) * weight
        return signal
    return 0.0


# SignalDetector에 현재가 기반 감지 메서드 추가
def add_realtime_detection():
    """SignalDetector에 실시간 감지 메서드 추가"""
    def detect_signal_with_price(self, current_price: float) -> dict:
        """
        현재가를 기준으로 매도 신호 감지 (장중 사용)
        
        RSI, 볼린저 밴드: 실시간 가격으로 재계산
        OBV, ATR: 일봉 데이터 그대로 사용 (종가 확정 필요)
        
        Args:
            current_price: 현재가
            
        Returns:
            신호 정보 딕셔너리
        """
        import math
        from sell_position_manager import SellPositionManager
        
        if self.data is None or self.data.empty:
            return None
        
        latest = self.data.iloc[-1]
        latest_date = latest['Date']
        
        # 저점 대비 보유일수
        low_idx = self.data['Low'].idxmin()
        low_date = self.data.loc[low_idx, 'Date']
        days_from_low = max(0, (latest_date - low_date).days)
        
        # 매수일 기준 보유 일수
        days_held = max(0, (latest_date - self.buy_date).days)
        
        # 수익률 계산 (현재가 기준)
        current_return = (current_price - self.buy_price) / self.buy_price
        
        # =====================================================================
        # 실시간 지표: RSI, 볼린저 밴드 (현재가로 재계산)
        # =====================================================================
        rsi_signal_value = generate_realtime_rsi_signal(
            self.data,
            current_price,
            self.params.get('rsi_overbought', 70),
            self.params.get('rsi_period', 14),
            self.params.get('rsi_weight', 1.0)
        )
        
        bb_signal_value = generate_realtime_bb_signal(
            self.data,
            current_price,
            self.params.get('bb_position_threshold', 100),
            self.params.get('bb_period', 20),
            self.params.get('bb_std_dev', 2.0),
            self.params.get('bb_weight', 1.0)
        )
        
        # =====================================================================
        # 일봉 지표: OBV, ATR (종가 기준, 실시간 재계산 불필요)
        # =====================================================================
        obv_signal = self.signal_generator.generate_obv_signal(
            self.params.get('obv_period', 20),
            self.params.get('obv_weight', 1.0)
        )
        obv_signal_value = obv_signal.iloc[-1] if not obv_signal.empty else 0
        
        atr_signal = self.signal_generator.generate_atr_signal(
            self.params.get('atr_multiplier', 2.0),
            self.params.get('atr_weight', 1.0)
        )
        atr_signal_value = atr_signal.iloc[-1] if not atr_signal.empty else 0
        
        # =====================================================================
        # 종합 신호 강도 계산 (EMA 제외)
        # =====================================================================
        signal_strength = (
            rsi_signal_value +
            obv_signal_value +
            atr_signal_value +
            bb_signal_value
        )
        
        # 신호 발생 지표 확인
        signal_indicators = []
        if rsi_signal_value > 0:
            signal_indicators.append('RSI')
        if obv_signal_value > 0:
            signal_indicators.append('OBV')
        if atr_signal_value > 0:
            signal_indicators.append('ATR')
        if bb_signal_value > 0:
            signal_indicators.append('BB')
        
        # 최대 가능 점수 (EMA 제외)
        max_possible_score = (
            self.params.get('rsi_weight', 1.0) +
            self.params.get('obv_weight', 1.0) +
            self.params.get('atr_weight', 1.0) +
            self.params.get('bb_weight', 1.0)
        )
        
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
        
        price_weight_exponent = self.params.get('price_weight_exponent', 2.0)
        price_weight = max(1.0, (1.0 + current_return) ** price_weight_exponent)
        
        time_weight_max = self.params.get('time_weight_max', 2.0)
        time_weight_midpoint = self.params.get('time_weight_midpoint', 365)
        time_weight_slope = self.params.get('time_weight_slope', 0.025)
        sigmoid = 1 / (1 + math.exp(-time_weight_slope * (days_held - time_weight_midpoint)))
        time_weight = 1 + (time_weight_max - 1) * sigmoid
        
        # 매도 비율 계산
        max_sell_ratio = self.params.get('max_sell_ratio', 0.01)
        current_position_ratio = 1.0
        
        if signal_strength > 0 and current_return >= 1.0:  # 수익률 100% 이상
            total_sell_ratio = max_sell_ratio * normalized_score * sell_weight * current_position_ratio * price_weight * time_weight
            total_sell_ratio = min(total_sell_ratio, current_position_ratio)
            
            min_sell_ratio = current_position_ratio * 0.05
            if total_sell_ratio < min_sell_ratio:
                total_sell_ratio = 0.0
        else:
            total_sell_ratio = 0.0
        
        hold_based_sell_ratio = total_sell_ratio / current_position_ratio if current_position_ratio > 0 else 0
        has_valid_signal = total_sell_ratio > 0
        
        # 실시간 RSI와 BB Position 값도 반환 (디버깅/참고용)
        realtime_rsi = calculate_realtime_rsi(
            self.data, current_price, self.params.get('rsi_period', 14)
        )
        realtime_bb_position = calculate_realtime_bb_position(
            self.data, current_price, 
            self.params.get('bb_period', 20),
            self.params.get('bb_std_dev', 2.0)
        )
        
        return {
            'date': latest_date,
            'price': current_price,
            'days_from_low': days_from_low,
            'days_held': days_held,
            'signal_strength': signal_strength,
            'signal_indicators': signal_indicators,
            'normalized_score': normalized_score,
            'sell_weight': sell_weight,
            'price_weight': price_weight,
            'time_weight': time_weight,
            'total_sell_ratio': total_sell_ratio,
            'hold_based_sell_ratio': hold_based_sell_ratio,
            'max_possible_score': max_possible_score,
            'has_signal': has_valid_signal,
            # 실시간 지표 값 (참고용)
            'realtime_rsi': realtime_rsi,
            'realtime_bb_position': realtime_bb_position
        }
    
    # SignalDetector 클래스에 메서드 추가
    SignalDetector.detect_signal_with_price = detect_signal_with_price

# 모듈 로드 시 메서드 추가
add_realtime_detection()


def main():
    """메인 함수"""
    import argparse
    
    # 명령줄 인자 파싱 (GitHub Actions 등 자동화 환경용)
    parser = argparse.ArgumentParser(description='매도 신호 알림 시스템')
    parser.add_argument('--mode', type=str, choices=['run_once', 'run_realtime', 'test', 'summary'],
                        help='실행 모드: run_once(한 번 체크), run_realtime(실시간), test(연결 테스트), summary(일일 요약)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔔 매도 신호 알림 시스템 (실시간 모니터링)")
    print("=" * 60)
    
    # 설정 확인
    if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("\n❌ 텔레그램 설정이 필요합니다!")
        print("\n📋 설정 방법:")
        print("1. config.py 파일을 열어주세요")
        print("2. TELEGRAM_BOT_TOKEN에 봇 토큰 입력")
        print("3. TELEGRAM_CHAT_ID에 채팅 ID 입력")
        return
    
    # 명령줄 인자로 모드가 지정된 경우 (자동화 환경)
    if args.mode:
        system = RealtimeSellSignalSystem()
        
        if args.mode == 'run_once':
            print("\n📊 GitHub Actions 모드: 한 번 체크")
            system.run_once()
        elif args.mode == 'run_realtime':
            system.run_realtime()
        elif args.mode == 'test':
            print("\n🧪 텔레그램 연결 테스트...")
            notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
            if notifier.test_connection():
                notifier.send_message(f"✅ 테스트 성공!\n\n시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        elif args.mode == 'summary':
            print("\n📊 일일 요약 전송...")
            system.send_daily_summary()
        return
    
    # 대화형 모드 (로컬 실행)
    print("\n실행 모드를 선택하세요:")
    print("1. 한 번 체크 후 종료")
    print("2. 실시간 모니터링 (매분 체크, 신호 시에만 알림)")
    print("3. 텔레그램 연결 테스트")
    print("4. 일일 요약 전송 테스트")
    
    try:
        choice = input("\n선택 (1/2/3/4): ").strip()
    except:
        choice = "1"
    
    system = RealtimeSellSignalSystem()
    
    if choice == "1":
        system.run_once()
    elif choice == "2":
        system.run_realtime()
    elif choice == "3":
        print("\n🧪 텔레그램 연결 테스트...")
        notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        if notifier.test_connection():
            notifier.send_message(f"✅ 테스트 성공!\n\n시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    elif choice == "4":
        print("\n📊 일일 요약 테스트...")
        system.send_daily_summary()
    else:
        print("❌ 잘못된 선택입니다.")


if __name__ == "__main__":
    main()
