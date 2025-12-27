# =============================================================================
# 실시간 데이터 수집 모듈 (Yahoo Finance API with Cookie/Crumb)
# =============================================================================
import pandas as pd
import requests
from datetime import datetime, timedelta
import re
from technical_indicators import TechnicalIndicators

# SSL 검증 비활성화 경고 무시
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class DataFetcher:
    """실시간 데이터 수집 클래스 (Yahoo Finance API)"""
    
    def __init__(self, ticker: str, start_date: str = None):
        """
        Args:
            ticker: Yahoo Finance 티커 심볼 (예: 'SOXL', 'USD')
            start_date: 데이터 시작일 (YYYY-MM-DD 형식)
        """
        self.ticker = ticker
        self.start_date = start_date or '2025-04-11'
        self.session = requests.Session()
        self.session.verify = False
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
    def _get_crumb_and_cookies(self):
        """Yahoo Finance에서 crumb과 쿠키 가져오기"""
        try:
            # 먼저 Yahoo Finance 페이지 방문하여 쿠키 획득
            url = f"https://finance.yahoo.com/quote/{self.ticker}/history"
            response = self.session.get(url, timeout=30)
            
            # crumb 추출 (여러 패턴 시도)
            patterns = [
                r'"crumb":"([^"]+)"',
                r'CrsyProvider.value = "([^"]+)"',
                r'"crumb":\s*"([^"]+)"'
            ]
            
            crumb = None
            for pattern in patterns:
                match = re.search(pattern, response.text)
                if match:
                    crumb = match.group(1)
                    break
            
            return crumb
            
        except Exception as e:
            print(f"⚠️ Crumb 획득 실패: {e}")
            return None
    
    def fetch_data_v8_api(self) -> pd.DataFrame:
        """Yahoo Finance v8 API 사용 (더 안정적)"""
        try:
            # 날짜 계산
            start_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
            end_dt = datetime.now() + timedelta(days=1)
            
            period1 = int(start_dt.timestamp())
            period2 = int(end_dt.timestamp())
            
            # v8 chart API 사용
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{self.ticker}"
            params = {
                'period1': period1,
                'period2': period2,
                'interval': '1d',
                'includePrePost': 'false',
                'events': 'div,splits'
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            
            # 데이터 파싱
            chart = data.get('chart', {}).get('result', [])
            if not chart:
                return None
            
            chart = chart[0]
            timestamps = chart.get('timestamp', [])
            quote = chart.get('indicators', {}).get('quote', [{}])[0]
            
            if not timestamps:
                return None
            
            df = pd.DataFrame({
                'Date': pd.to_datetime(timestamps, unit='s'),
                'Open': quote.get('open', []),
                'High': quote.get('high', []),
                'Low': quote.get('low', []),
                'Close': quote.get('close', []),
                'Volume': quote.get('volume', [])
            })
            
            # NaN 제거
            df = df.dropna()
            
            return df
            
        except Exception as e:
            print(f"⚠️ v8 API 오류: {e}")
            return None
        
    def fetch_data(self) -> pd.DataFrame:
        """
        Yahoo Finance에서 데이터를 가져와 기술적 지표 추가
        
        Returns:
            기술적 지표가 포함된 DataFrame
        """
        try:
            print(f"📡 {self.ticker} 데이터 다운로드 중...")
            
            end_dt = datetime.now() + timedelta(days=1)
            print(f"   기간: {self.start_date} ~ {end_dt.strftime('%Y-%m-%d')}")
            
            # v8 API 시도
            data = self.fetch_data_v8_api()
            
            if data is None or data.empty:
                print(f"❌ {self.ticker}: 데이터를 가져올 수 없습니다.")
                return None
            
            print(f"✅ {len(data)}일치 데이터 로드 완료")
            print(f"   최신 날짜: {data['Date'].max().strftime('%Y-%m-%d')}")
            print(f"   최신 종가: {data['Close'].iloc[-1]:.2f}")
            
            # 기술적 지표 추가
            ti = TechnicalIndicators(data)
            ti.calculate_all_indicators()
            data_with_indicators = ti.data  # 지표가 추가된 데이터
            
            return data_with_indicators
            
        except Exception as e:
            print(f"❌ 데이터 수집 오류: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_latest_data(self) -> dict:
        """
        최신 데이터 및 정보 반환
        
        Returns:
            최신 데이터 딕셔너리
        """
        data = self.fetch_data()
        if data is None or data.empty:
            return None
        
        latest = data.iloc[-1]
        
        return {
            'date': latest['Date'],
            'close': latest['Close'],
            'high': latest['High'],
            'low': latest['Low'],
            'open': latest['Open'],
            'volume': latest.get('Volume', 0),
            'data': data  # 전체 데이터프레임도 포함
        }
    
    def get_realtime_price(self) -> dict:
        """
        실시간 현재가 가져오기 (장중 사용)
        
        Returns:
            실시간 가격 정보 딕셔너리
        """
        try:
            # v8 chart API로 최신 데이터 가져오기 (1분봉)
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{self.ticker}"
            params = {
                'interval': '1m',
                'range': '1d',
                'includePrePost': 'true'  # 프리/애프터 마켓 포함
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            chart = data.get('chart', {}).get('result', [])
            
            if not chart:
                return None
            
            chart = chart[0]
            meta = chart.get('meta', {})
            
            # 현재가 (regularMarketPrice 또는 최신 close)
            current_price = meta.get('regularMarketPrice')
            
            if current_price is None:
                # 최신 분봉에서 가져오기
                timestamps = chart.get('timestamp', [])
                quote = chart.get('indicators', {}).get('quote', [{}])[0]
                closes = quote.get('close', [])
                
                # None이 아닌 마지막 값 찾기
                for i in range(len(closes) - 1, -1, -1):
                    if closes[i] is not None:
                        current_price = closes[i]
                        break
            
            if current_price is None:
                return None
            
            return {
                'ticker': self.ticker,
                'price': current_price,
                'timestamp': datetime.now(),
                'market_state': meta.get('marketState', 'UNKNOWN'),  # PRE, REGULAR, POST, CLOSED
                'previous_close': meta.get('previousClose'),
                'change_percent': ((current_price - meta.get('previousClose', current_price)) / meta.get('previousClose', current_price) * 100) if meta.get('previousClose') else 0
            }
            
        except Exception as e:
            print(f"⚠️ 실시간 가격 조회 실패 ({self.ticker}): {e}")
            return None
    
    def fetch_intraday_data(self, interval: str = '5m') -> pd.DataFrame:
        """
        장중 데이터 가져오기 (분봉)
        
        Args:
            interval: 봉 간격 ('1m', '5m', '15m', '30m', '1h')
        
        Returns:
            분봉 데이터 DataFrame
        """
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{self.ticker}"
            params = {
                'interval': interval,
                'range': '1d',
                'includePrePost': 'true'
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            chart = data.get('chart', {}).get('result', [])
            
            if not chart:
                return None
            
            chart = chart[0]
            timestamps = chart.get('timestamp', [])
            quote = chart.get('indicators', {}).get('quote', [{}])[0]
            
            if not timestamps:
                return None
            
            df = pd.DataFrame({
                'Date': pd.to_datetime(timestamps, unit='s'),
                'Open': quote.get('open', []),
                'High': quote.get('high', []),
                'Low': quote.get('low', []),
                'Close': quote.get('close', []),
                'Volume': quote.get('volume', [])
            })
            
            # NaN 제거
            df = df.dropna()
            
            return df
            
        except Exception as e:
            print(f"⚠️ 장중 데이터 조회 실패: {e}")
            return None


def test_fetch():
    """데이터 수집 테스트"""
    print("=" * 60)
    print("데이터 수집 테스트")
    print("=" * 60)
    
    # SOXL 테스트
    print("\n[SOXL 테스트]")
    soxl_fetcher = DataFetcher('SOXL', '2025-04-11')
    soxl_data = soxl_fetcher.get_latest_data()
    if soxl_data:
        print(f"최신 날짜: {soxl_data['date']}")
        print(f"최신 종가: ${soxl_data['close']:.2f}")
    else:
        print("❌ SOXL 데이터 수집 실패")
    
    # USD 테스트
    print("\n[USD 테스트]")
    usd_fetcher = DataFetcher('USD', '2025-04-11')
    usd_data = usd_fetcher.get_latest_data()
    if usd_data:
        print(f"최신 날짜: {usd_data['date']}")
        print(f"최신 종가: ${usd_data['close']:.2f}")
    else:
        print("❌ USD 데이터 수집 실패")


if __name__ == "__main__":
    test_fetch()
