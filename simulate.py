# =============================================================================
# 매도 신호 시뮬레이션
# 날짜와 주가를 입력하여 매도 신호 유무를 확인
#
# 파라미터 구조:
#   --date     : 기준날짜. 신호(RSI/BB 등) + sell_count + 수익률가격 + 시간 모두 이 날짜 기준
#   --price    : (선택) 수익률 가중치 기준가격만 별도 지정. 신호 감지는 --date 실제 종가 유지
#   --time-date: (선택) 시간 가중치 기준날짜만 별도 지정. days_held가 이 날짜 기준으로 변경
#
# 사용 예:
#   # 기본: 모든 기준을 6/5 실제 데이터로
#   python simulate.py --symbol SOXL --date 2026-06-05
#
#   # 신호/sell_count는 6/5 유지, 시간만 9/1 기준으로 (시간 가중치 변화 확인)
#   python simulate.py --symbol SOXL --date 2026-06-05 --time-date 2026-09-01
#
#   # 신호/sell_count/시간은 6/5, 수익률 가중치만 $300 기준으로
#   python simulate.py --symbol SOXL --date 2026-06-05 --price 300
#
#   # 미래날짜: 현재가를 신호용 가격으로 자동 조회, 시간은 7/1 기준
#   python simulate.py --symbol SOXL --date 2026-07-01 --auto-price
#
#   # USD 종목
#   python simulate.py --symbol USD --date 2026-06-05
#
# 복수 날짜 시뮬레이션 (스크립트 하단 MULTI_DATE_ENTRIES 수정):
#   python simulate.py --multi
# =============================================================================
import sys
import os
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import DataFetcher
from technical_indicators import TechnicalIndicators
from sell_signal_generator import SellSignalGenerator
from signal_detector import SignalDetector
from config import SYMBOLS

# =============================================================================
# 복수 날짜 시뮬레이션 설정 (--multi 옵션 사용 시)
# 날짜 순서대로 입력 - 이전 날짜의 매도가 다음 날짜의 sell_count에 누적됩니다.
# 'price': 생략하면 해당 날짜 실제 종가 자동 조회. 미래날짜는 반드시 입력.
# =============================================================================
MULTI_DATE_ENTRIES = {
    'SOXL': [
        {'date': '2026-07-01', 'price': 200.0},
        {'date': '2026-07-15', 'price': 220.0},
        {'date': '2026-07-31', 'price': 240.0},
    ],
    'USD': [
        {'date': '2026-07-01', 'price': 100.0},
    ],
}


def _fetch_raw(ticker: str, buy_date: str) -> pd.DataFrame:
    """Yahoo Finance에서 전체 raw OHLCV 데이터 수집 (날짜 필터 없음)"""
    fetcher = DataFetcher(ticker, buy_date)
    raw = fetcher.fetch_data_v8_api()
    if raw is None or raw.empty:
        raise RuntimeError(f"{ticker} 데이터 수집 실패")
    raw['Date'] = pd.to_datetime(raw['Date']).dt.tz_localize(None)
    return raw


def _resolve_buy_price(hist: pd.DataFrame, buy_date: str, buy_price_override: float) -> float:
    """
    매수 평균가 결정.
    config에 buy_price가 None이면 buy_date 종가 × 2 사용 (signal_detector와 동일 로직).
    """
    if buy_price_override is not None:
        return buy_price_override

    buy_dt = pd.to_datetime(buy_date)
    buy_day_data = hist[hist['Date'] >= buy_dt]
    if not buy_day_data.empty:
        return float(buy_day_data.iloc[0]['Close']) * 2.0
    return float(hist['Close'].iloc[0]) * 2.0


def compute_historical_sell_count(
    hist: pd.DataFrame,
    params: dict,
    buy_price: float,
    warmup: int = 5,
) -> int:
    """
    백테스트 엔진과 동일한 방식으로 sell_count 계산.

    backtest_engine.py 로직:
      - 신호 발생(signal_strength > 0) + 수익률 >= 100% 인 날마다 signal_count 증가
      - 처음 warmup(5)번은 카운트만 하고 매도 안 함
      - 그 이후부터 실제 매도 여부와 무관하게 position_manager.sell_count 증가

    Returns:
        warmup 이후 누적 신호 발생 횟수 (= 백테스트의 position_manager.sell_count 근사값)
    """
    ti = TechnicalIndicators(hist)
    generator = SellSignalGenerator(ti.data)
    combined_signal = generator.generate_combined_signal(params)

    signal_count = 0

    for i in range(len(hist)):
        price = float(hist['Close'].iloc[i])
        current_return = (price - buy_price) / buy_price
        score = float(combined_signal.iloc[i])

        if score > 0 and current_return >= 1.0:
            signal_count += 1

    effective = max(0, signal_count - warmup)
    print(f"  신호 발생일 합계: {signal_count}회  /  워밍업({warmup}) 제외 sell_count: {effective}")
    return effective


def simulate_sell_signal(
    symbol_key: str,
    sim_date: str,
    price_override: float = None,
    high: float = None,
    low: float = None,
    open_: float = None,
    volume: float = 0,
    extra_sell_count: int = 0,
    buy_price_override: float = None,
    time_date: str = None,
) -> dict:
    """
    특정 날짜/주가를 가정한 매도 신호 시뮬레이션.

    파라미터 역할:
        sim_date      : [신호 + sell_count + 기본 time] 기준날짜.
                        이 날짜의 실제 종가를 Yahoo Finance에서 자동 조회해
                        신호 감지용 시뮬레이션 행에 사용.
        price_override: [수익률 가중치 전용] 지정 시 이 가격으로만 수익률 가중치 계산.
                        신호 감지(RSI/BB 등)는 sim_date 실제 종가를 그대로 사용.
                        None이면 sim_date 실제 종가를 수익률 가중치에도 사용.
        time_date     : [시간 가중치 전용] 지정 시 days_held를 이 날짜 기준으로 계산.
                        None이면 sim_date 기준.

    Returns:
        SignalDetector.detect_signal() 결과 딕셔너리
    """
    symbol_config = SYMBOLS[symbol_key]
    params = symbol_config['params']
    buy_date = symbol_config['buy_date']

    # 1. 전체 raw 데이터 수집 (sim_date 포함 가능)
    print(f"  [{symbol_config['ticker']}] 데이터 수집 중 ({buy_date} ~ {sim_date})...")
    raw = _fetch_raw(symbol_config['ticker'], buy_date)
    sim_dt = pd.to_datetime(sim_date)

    # 2. sim_date 실제 종가 조회 (신호 감지용 close)
    # Yahoo Finance 일봉 timestamps have time components (e.g. 13:30:00 UTC)
    # so we compare .dt.date (date only) to avoid midnight vs 13:30 mismatch
    sim_row_data = raw[raw['Date'].dt.date == sim_dt.date()]
    if not sim_row_data.empty:
        actual_close = float(sim_row_data.iloc[0]['Close'])
        print(f"  실제 종가 ({sim_date}): ${actual_close:.2f}")
    else:
        # 미래날짜 또는 주말/공휴일 - price_override 필수
        if price_override is None:
            raise RuntimeError(
                f"{sim_date} 실제 데이터 없음. --price 로 가격을 직접 입력하세요."
            )
        actual_close = price_override
        print(f"  [{sim_date}] 실제 데이터 없음 - --price ${actual_close:.2f} 를 신호용 가격으로 사용")

    # 3. hist = sim_date 이전 데이터 (기술적 지표 계산용)
    hist = raw[raw['Date'] < sim_dt].copy().reset_index(drop=True)
    if hist.empty:
        raise RuntimeError(f"{sim_date} 이전 데이터가 없습니다")
    print(f"  {len(hist)}일 로드 완료 (최신: {hist['Date'].iloc[-1].strftime('%Y-%m-%d')})")

    # 4. 매수 평균가 결정
    buy_price = _resolve_buy_price(hist, buy_date, buy_price_override)
    print(f"  매수 평균가: ${buy_price:.2f}")

    # 5. 수익률 가중치 기준가격 결정
    price_for_weight = price_override if price_override is not None else actual_close
    if price_override is not None and abs(price_override - actual_close) > 0.01:
        ret_actual = (actual_close - buy_price) / buy_price * 100
        ret_price  = (price_override - buy_price) / buy_price * 100
        print(f"  수익률 기준가 (--price): ${price_for_weight:.2f} ({ret_price:+.2f}%)  "
              f"[신호용: ${actual_close:.2f} ({ret_actual:+.2f}%)]")

    # 6. 백테스트 방식으로 sell_count 자동 계산 (--date 기준)
    historical_sell_count = compute_historical_sell_count(hist, params, buy_price)
    total_sell_count = historical_sell_count + extra_sell_count

    # 7. 거래량 기본값: 최근 20일 평균
    avg_volume = hist['Volume'].tail(20).mean() if volume <= 0 else volume

    # 8. 시뮬레이션 행 추가
    #    --time-date 지정 시 해당 날짜를 Date로 사용 → days_held 변경
    effective_date = pd.to_datetime(time_date) if time_date is not None else sim_dt
    if time_date is not None:
        days_diff = (effective_date - sim_dt).days
        print(f"  시간 가중치 기준날짜: {time_date} (days_held +{days_diff}일)")

    sim_row = pd.DataFrame([{
        'Date':   effective_date,
        'Open':   open_ if open_ is not None else actual_close,
        'High':   high  if high  is not None else actual_close,
        'Low':    low   if low   is not None else actual_close,
        'Close':  actual_close,  # 항상 실제 종가 사용 (신호 감지용)
        'Volume': avg_volume,
    }])

    combined = pd.concat([hist, sim_row], ignore_index=True)

    # 9. 기술적 지표 재계산
    ti = TechnicalIndicators(combined)
    data_with_indicators = ti.data

    # 10. 매도 신호 감지
    detector = SignalDetector(
        data_with_indicators, params, buy_date, buy_price, sell_count=total_sell_count
    )
    result = detector.detect_signal(price_for_weight=price_for_weight)

    if result is not None:
        result['computed_sell_count'] = total_sell_count
        result['sim_actual_close'] = actual_close

    return result


def print_result(symbol_key: str, sim_date: str, signal: dict, buy_price: float = None):
    """시뮬레이션 결과 출력"""
    symbol_config = SYMBOLS[symbol_key]

    print(f"\n{'='*60}")
    print(f"  매도 신호 시뮬레이션 결과: {symbol_config['name']}")
    print(f"{'='*60}")
    print(f"  날짜    : {sim_date}")

    if signal is None:
        print("\n  [오류] 신호 계산 실패")
        print(f"{'='*60}")
        return

    actual_close   = signal.get('sim_actual_close', signal.get('price', 0))
    price_for_wt   = signal.get('price_for_weight', actual_close)
    has_price_diff = abs(price_for_wt - actual_close) > 0.01

    print(f"  신호용 종가  : ${actual_close:.2f}")
    if has_price_diff:
        print(f"  수익률 기준가: ${price_for_wt:.2f}  (--price 지정)")

    if buy_price:
        ret = (price_for_wt - buy_price) / buy_price * 100
        print(f"  매수 평균가  : ${buy_price:.2f}  |  수익률: {ret:+.2f}%")

    print(f"  sell_count: {signal.get('computed_sell_count', '?')} (백테스트 방식 자동 계산)")

    indicators = signal.get('signal_indicators', [])
    print(f"\n  신호 지표   : {' + '.join(indicators) if indicators else '없음'}")
    print(f"  신호 강도   : {signal.get('signal_strength', 0):.4f} / "
          f"{signal.get('max_possible_score', 0):.0f}  "
          f"(정규화: {signal.get('normalized_score', 0):.4f})")
    print(f"  매도 가중치 : {signal.get('sell_weight', 0):.4f}")
    print(f"  수익률 가중치: {signal.get('price_weight', 0):.4f}"
          + (f"  (기준가 ${price_for_wt:.2f})" if has_price_diff else ""))
    print(f"  시간 가중치 : {signal.get('time_weight', 0):.4f}")
    print(f"  보유일수    : {signal.get('days_held', 0)}일")

    total_ratio = signal.get('total_sell_ratio', 0)
    raw_ratio   = signal.get('raw_sell_ratio', total_ratio)
    has_signal  = signal.get('has_signal', False)

    print(f"\n  [결과] {'매도 신호: 있음' if has_signal else '매도 신호: 없음'}")
    if has_signal:
        print(f"  전체 기준 매도 비율: {total_ratio * 100:.2f}%")
        print(f"  보유 기준 매도 비율: {signal.get('hold_based_sell_ratio', 0) * 100:.2f}%")
    else:
        print(f"  계산된 매도 비율  : {raw_ratio * 100:.4f}%  (5% 미만 -> 신호 없음)")
    print(f"{'='*60}")


def fetch_current_price(symbol_key: str) -> float:
    """Yahoo Finance에서 해당 종목의 최신 종가를 가져옵니다."""
    symbol_config = SYMBOLS[symbol_key]
    raw = _fetch_raw(symbol_config['ticker'], symbol_config['buy_date'])
    price = float(raw['Close'].iloc[-1])
    date  = pd.to_datetime(raw['Date'].iloc[-1]).strftime('%Y-%m-%d')
    print(f"  [{symbol_key}] 현재가 자동 조회: ${price:.2f}  (기준일: {date})")
    return price


def run_multi(symbol_key: str, buy_price_override: float = None):
    """
    MULTI_DATE_ENTRIES에 정의된 복수 날짜를 순서대로 시뮬레이션.
    이전 날짜에서 실제 매도가 발생하면 extra_sell_count에 누적됩니다.
    """
    entries = MULTI_DATE_ENTRIES.get(symbol_key, [])
    if not entries:
        print(f"[오류] MULTI_DATE_ENTRIES에 {symbol_key} 항목이 없습니다.")
        return

    extra_sell_count = 0

    for entry in entries:
        signal = simulate_sell_signal(
            symbol_key=symbol_key,
            sim_date=entry['date'],
            price_override=entry.get('price'),
            high=entry.get('high'),
            low=entry.get('low'),
            open_=entry.get('open'),
            volume=entry.get('volume', 0),
            extra_sell_count=extra_sell_count,
            buy_price_override=buy_price_override,
            time_date=entry.get('time_date'),
        )

        print_result(symbol_key, entry['date'], signal, buy_price_override)

        if signal and signal.get('has_signal'):
            extra_sell_count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='매도 신호 시뮬레이션',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--symbol',     choices=['SOXL', 'USD'], default='SOXL',
                        help='종목 (기본: SOXL)')
    parser.add_argument('--date',       help='기준날짜 (YYYY-MM-DD) - 신호/sell_count/기본 time 모두 적용')
    parser.add_argument('--price',      type=float,
                        help='수익률 가중치 기준가격 (미입력 시 --date 실제 종가 자동 사용)\n'
                             '신호 감지(RSI 등)는 --date 실제 종가 그대로 유지')
    parser.add_argument('--time-date',  help='시간 가중치 기준날짜 (YYYY-MM-DD)\n'
                                             '미입력 시 --date 기준. days_held만 변경됨')
    parser.add_argument('--buy-price',  type=float, help='매수 평균가 직접 지정')
    parser.add_argument('--high',       type=float, help='고가 (기본: 실제 종가)')
    parser.add_argument('--low',        type=float, help='저가 (기본: 실제 종가)')
    parser.add_argument('--open',       type=float, dest='open_', help='시가 (기본: 실제 종가)')
    parser.add_argument('--volume',     type=float, default=0, help='거래량 (기본: 최근 20일 평균)')
    parser.add_argument('--auto-price', action='store_true',
                        help='현재가 자동 조회 (미래날짜 시뮬레이션용)\n'
                             '신호용 가격으로 사용. --price 지정 시 수익률 가중치만 --price 사용')
    parser.add_argument('--multi',      action='store_true',
                        help='MULTI_DATE_ENTRIES 복수 날짜 시뮬레이션 실행')

    args = parser.parse_args()

    if args.multi:
        run_multi(args.symbol, args.buy_price)
    else:
        if not args.date:
            parser.error("--date 는 필수입니다.")

        # 미래날짜 시뮬레이션: --auto-price로 현재가를 신호용 가격으로 사용
        price_arg = args.price
        if args.auto_price:
            auto_fetched = fetch_current_price(args.symbol)
            # price_override가 없으면 auto_fetched를 신호+수익률 모두에 사용
            if price_arg is None:
                price_arg = auto_fetched

        signal = simulate_sell_signal(
            symbol_key=args.symbol,
            sim_date=args.date,
            price_override=price_arg,
            high=args.high,
            low=args.low,
            open_=args.open_,
            volume=args.volume,
            buy_price_override=args.buy_price,
            time_date=args.time_date,
        )
        print_result(args.symbol, args.date, signal, args.buy_price)
