# =============================================================================
# 매도 신호 알림 시스템 설정
# =============================================================================
import os

# -----------------------------------------------------------------------------
# 텔레그램 설정
# -----------------------------------------------------------------------------
# 1. @BotFather에서 봇 생성 후 토큰 입력
# 2. 봇과 대화 시작 후 @userinfobot에서 Chat ID 확인
# 환경변수가 있으면 환경변수 사용, 없으면 기본값 사용 (로컬 실행용)
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', "8594787714:AAGLuJXNFmjHQUAmFiIjcQoD9HnRPHgumHg")
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', "6860037461")

# -----------------------------------------------------------------------------
# SOXL 확정 파라미터 (2024-12-21 최종 결정)
# -----------------------------------------------------------------------------
SOXL_PARAMS = {
    # RSI 파라미터
    'rsi_overbought': 66,
    'rsi_period': 43,
    'rsi_weight': 9,
    
    # OBV 파라미터
    'obv_period': 57,
    'obv_weight': 3,
    
    # ATR 파라미터
    'atr_multiplier': 2,
    'atr_weight': 3,
    
    # 볼린저밴드 파라미터
    'bb_position_threshold': 100,
    'bb_period': 40,
    'bb_std_dev': 3.5,
    'bb_weight': 3,
    
    # EMA 파라미터
    'ema_period': 60,
    
    # 시스템 파라미터
    'max_sell_ratio': 0.01,
    'sell_weight_base': 1.01,
    'price_weight_exponent': 0.4,
    'sell_weight_coefficient': 1.6,
    
    # 시간 가중치 파라미터
    'time_weight_max': 40,
    'time_weight_midpoint': 550,
    'time_weight_slope': 0.045
}

# -----------------------------------------------------------------------------
# USD 확정 파라미터 (2024-12-21 최종 결정)
# -----------------------------------------------------------------------------
USD_PARAMS = {
    # RSI 파라미터
    'rsi_overbought': 65,
    'rsi_period': 48,
    'rsi_weight': 10,
    
    # OBV 파라미터
    'obv_period': 60,
    'obv_weight': 5.0,
    
    # ATR 파라미터
    'atr_multiplier': 3,
    'atr_weight': 4.0,
    
    # 볼린저밴드 파라미터
    'bb_position_threshold': 100,
    'bb_period': 60,
    'bb_std_dev': 2.0,
    'bb_weight': 6,
    
    # EMA 파라미터
    'ema_period': 60,
    
    # 시스템 파라미터
    'max_sell_ratio': 0.01,
    'sell_weight_base': 1.04,
    'price_weight_exponent': 2.0,
    'sell_weight_coefficient': 0.04,
    
    # 시간 가중치 파라미터 (비활성화)
    'time_weight_max': 1,
    'time_weight_midpoint': 1,
    'time_weight_slope': 0.045
}

# -----------------------------------------------------------------------------
# 종목 설정
# -----------------------------------------------------------------------------
SYMBOLS = {
    'SOXL': {
        'ticker': 'SOXL',           # yfinance 티커
        'name': 'SOXL (반도체 3배 레버리지)',
        'params': SOXL_PARAMS,
        'buy_date': '2025-04-21',   # 분석 시작일
        'buy_price': None,          # 매수 평균가 (None이면 시작일 종가의 2배 사용)
    },
    'USD': {
        'ticker': 'USD',            # ProShares Ultra Semiconductors
        'name': 'USD (반도체 2배 레버리지)',
        'params': USD_PARAMS,
        'buy_date': '2025-04-21',   # 분석 시작일
        'buy_price': None,          # 매수 평균가 (None이면 시작일 종가의 2배 사용)
    }
}

# -----------------------------------------------------------------------------
# 알림 설정
# -----------------------------------------------------------------------------
# 신호 강도 임계값 (이 값 이상일 때만 알림)
SIGNAL_THRESHOLD = 0.0  # 0이면 모든 매도 신호에 알림

# 알림 쿨다운 (같은 종목에 대해 연속 알림 방지, 시간 단위)
ALERT_COOLDOWN_HOURS = 24

# -----------------------------------------------------------------------------
# 스케줄 설정
# -----------------------------------------------------------------------------
# 체크 주기 (분) - 실시간 모니터링을 위해 1분으로 설정
CHECK_INTERVAL_MINUTES = 1

# 운영 시간 (한국 시간 기준)
# 미국 정규장: 23:30~06:00 (서머타임), 00:30~07:00 (표준시)
OPERATING_HOURS = {
    'start': 18,   # 시작 시간 (프리마켓 포함)
    'end': 10,      # 종료 시간 (애프터마켓 포함)
}

# 일일 요약 전송 시간 (한국 시간, 24시간 형식)
# 미국 장 마감 후: 오전 6시 (서머타임), 오전 7시 (표준시)
DAILY_SUMMARY_TIME = {
    'hour': 7,
    'minute': 0
}

# 미국 장 시간 (한국 시간 기준, 서머타임 기준)
US_MARKET_HOURS = {
    'pre_market_start': 18,    # 프리마켓 시작 (22:00)
    'regular_start': 23,       # 정규장 시작 (23:30)
    'regular_end': 7,          # 정규장 마감 (06:00)
    'after_market_end': 10      # 애프터마켓 마감 (08:00)
}

