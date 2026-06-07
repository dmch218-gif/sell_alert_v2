# =============================================================================
# 4/21부터 현재까지 매도 신호 확인 스크립트
# =============================================================================
import pandas as pd
from datetime import datetime
from data_fetcher import DataFetcher
from signal_detector import SignalDetector
from config import SYMBOLS

def check_historical_signals(symbol_key: str, symbol_config: dict):
    """
    과거 매수일(buy_date)부터 현재까지의 종가 기준 매도 신호를 모두 확인
    """
    print(f"\n{'='*60}")
    print(f"📊 {symbol_config['name']} 매도 신호 분석")
    print(f"{'='*60}")
    
    # 데이터 가져오기
    fetcher = DataFetcher(symbol_config['ticker'], symbol_config['buy_date'])
    full_data_info = fetcher.get_historical_data()
    
    if full_data_info is None:
        print("❌ 데이터를 가져올 수 없습니다.")
        return
    
    historical_data = full_data_info['data']
    
    # 매수일 이후 데이터만 필터링
    buy_date_dt = pd.to_datetime(symbol_config['buy_date'])
    historical_data = historical_data[historical_data['Date'] >= buy_date_dt].copy()
    
    if historical_data.empty:
        print("❌ 매수일 이후 데이터가 없습니다.")
        return
    
    print(f"📅 분석 기간: {symbol_config['buy_date']} ~ {historical_data['Date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"📈 총 {len(historical_data)}일 데이터")
    
    # 매수가 설정 (None이면 시작일 종가의 2배)
    buy_price = symbol_config['buy_price']
    if buy_price is None:
        first_close = historical_data.iloc[0]['Close']
        buy_price = first_close * 2.0
        print(f"💰 매수가: ${buy_price:.2f} (시작일 종가 ${first_close:.2f}의 2배)")
    else:
        print(f"💰 매수가: ${buy_price:.2f}")
    
    params = symbol_config['params']
    
    # 최대 가능 신호강도 (EMA 제외)
    max_possible_score = (
        params.get('rsi_weight', 1.0) +
        params.get('obv_weight', 1.0) +
        params.get('atr_weight', 1.0) +
        params.get('bb_weight', 1.0)
    )
    print(f"📊 최대 신호강도: {max_possible_score}")
    print(f"🎯 수익률 100% 이상 조건 적용")
    
    historical_sell_events = []
    cumulative_ratio = 0.0
    temp_sell_count = 0
    
    # 필요한 최소 데이터 기간
    min_period = max(
        params.get('rsi_period', 14),
        params.get('obv_period', 20),
        params.get('bb_period', 20),
        params.get('atr_period', 14)
    )
    
    print(f"\n{'─'*60}")
    print("🔍 매도 신호 검색 중...")
    print(f"{'─'*60}")
    
    for i in range(len(historical_data)):
        daily_data = historical_data.iloc[:i+1].copy()
        
        if len(daily_data) < min_period:
            continue
        
        # SignalDetector 생성
        detector = SignalDetector(
            daily_data,
            params,
            symbol_config['buy_date'],
            buy_price,
            sell_count=temp_sell_count
        )
        
        signal = detector.detect_signal()
        
        if signal and signal['has_signal'] and signal['total_sell_ratio'] > 0:
            current_return = (signal['price'] - buy_price) / buy_price
            
            # 수익률 100% 미만이면 스킵
            if current_return < 1.0:
                continue
            
            temp_sell_count += 1
            cumulative_ratio += signal['total_sell_ratio']
            
            event = {
                'no': temp_sell_count,
                'date': signal['date'].strftime('%Y-%m-%d'),
                'price': signal['price'],
                'return': current_return * 100,
                'signal_strength': signal['signal_strength'],
                'signal_indicators': signal['signal_indicators'],
                'sell_ratio': signal['total_sell_ratio'] * 100,
                'cumulative_ratio': cumulative_ratio * 100,
                'remaining_ratio': max(0, (1.0 - cumulative_ratio)) * 100
            }
            historical_sell_events.append(event)
            
            print(f"\n✅ 매도 신호 #{temp_sell_count}")
            print(f"   날짜: {event['date']}")
            print(f"   가격: ${event['price']:.2f}")
            print(f"   수익률: {event['return']:.2f}%")
            print(f"   신호강도: {event['signal_strength']:.4f}")
            print(f"   신호지표: {event['signal_indicators']}")
            print(f"   매도비율: {event['sell_ratio']:.2f}%")
            print(f"   누적매도: {event['cumulative_ratio']:.2f}%")
            print(f"   잔여비중: {event['remaining_ratio']:.2f}%")
    
    print(f"\n{'='*60}")
    print(f"📊 {symbol_config['name']} 분석 결과 요약")
    print(f"{'='*60}")
    
    if historical_sell_events:
        print(f"\n🎯 총 매도 신호: {len(historical_sell_events)}회")
        print(f"📉 총 매도비율: {cumulative_ratio*100:.2f}%")
        print(f"📊 잔여 보유비율: {(1-cumulative_ratio)*100:.2f}%")
        
        print(f"\n{'─'*60}")
        print("📋 전체 매도 이력")
        print(f"{'─'*60}")
        print(f"{'No':<3} {'날짜':<12} {'가격':>10} {'수익률':>8} {'신호강도':>8} {'매도%':>7} {'누적%':>7} {'잔여%':>7}")
        print(f"{'─'*60}")
        for e in historical_sell_events:
            print(f"{e['no']:<3} {e['date']:<12} ${e['price']:>8.2f} {e['return']:>7.1f}% {e['signal_strength']:>8.4f} {e['sell_ratio']:>6.2f}% {e['cumulative_ratio']:>6.2f}% {e['remaining_ratio']:>6.2f}%")
    else:
        print("\n❌ 매도 신호 없음")
        print("   - 수익률 100% 이상 조건을 충족하는 신호가 없었습니다.")
        
        # 추가 분석: 현재 수익률 확인
        current_price = historical_data.iloc[-1]['Close']
        current_return = (current_price - buy_price) / buy_price * 100
        print(f"\n📈 현재 상태:")
        print(f"   - 현재가: ${current_price:.2f}")
        print(f"   - 현재 수익률: {current_return:.2f}%")
        print(f"   - 매도 조건: 수익률 100% 이상 필요")
    
    return historical_sell_events


if __name__ == "__main__":
    print("\n" + "="*60)
    print("📊 4/21 ~ 현재 매도 신호 분석 시작")
    print("="*60)
    
    # SOXL 분석
    soxl_signals = check_historical_signals('SOXL', SYMBOLS['SOXL'])
    
    # USD 분석
    usd_signals = check_historical_signals('USD', SYMBOLS['USD'])
    
    print("\n" + "="*60)
    print("✅ 분석 완료")
    print("="*60)
