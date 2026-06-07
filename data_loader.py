# =============================================================================
# 데이터 로더 및 전처리 모듈
# =============================================================================
import pandas as pd
import numpy as np
from datetime import datetime

class DataLoader:
    """SOXL 데이터 로더 및 전처리 클래스"""
    
    def __init__(self, csv_path):
        """
        Args:
            csv_path: CSV 파일 경로
        """
        self.csv_path = csv_path
        self.data = None
    
    def load_data(self):
        """CSV 파일 로드 및 전처리"""
        try:
            # CSV 파일 로드
            self.data = pd.read_csv(self.csv_path)
            
            # 날짜 컬럼 처리
            if 'Date' in self.data.columns:
                self.data['Date'] = pd.to_datetime(self.data['Date'], format='%m/%d/%Y', errors='coerce')
                self.data.set_index('Date', inplace=True)
            elif 'date' in self.data.columns:
                self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')
                self.data.set_index('date', inplace=True)
            
            # 컬럼명 정규화 (Price -> Close)
            if 'Price' in self.data.columns:
                self.data['Close'] = self.data['Price']
            if 'Open' in self.data.columns:
                self.data['Open'] = self.data['Open']
            if 'High' in self.data.columns:
                self.data['High'] = self.data['High']
            if 'Low' in self.data.columns:
                self.data['Low'] = self.data['Low']
            
            # 거래량 처리 (M, K 등 제거)
            if 'Vol.' in self.data.columns:
                self.data['Volume'] = self.data['Vol.'].apply(self._parse_volume)
            elif 'Volume' in self.data.columns:
                self.data['Volume'] = self.data['Volume'].apply(self._parse_volume)
            else:
                self.data['Volume'] = 0
            
            # 필요한 컬럼만 선택
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")
            
            # 데이터 정렬 (날짜순)
            self.data = self.data.sort_index()
            
            # 결측치 제거
            self.data = self.data.dropna(subset=['Open', 'High', 'Low', 'Close'])
            
            return self.data
            
        except Exception as e:
            print(f"❌ 데이터 로드 중 오류 발생: {e}")
            raise
    
    def _parse_volume(self, vol_str):
        """거래량 문자열 파싱 (예: '63.20M' -> 63200000)"""
        if pd.isna(vol_str) or vol_str == '':
            return 0
        
        try:
            vol_str = str(vol_str).strip()
            if 'M' in vol_str.upper():
                return float(vol_str.replace('M', '').replace(',', '')) * 1_000_000
            elif 'K' in vol_str.upper():
                return float(vol_str.replace('K', '').replace(',', '')) * 1_000
            elif 'B' in vol_str.upper():
                return float(vol_str.replace('B', '').replace(',', '')) * 1_000_000_000
            else:
                return float(vol_str.replace(',', ''))
        except:
            return 0
    
    def get_data(self):
        """전처리된 데이터 반환"""
        if self.data is None:
            raise ValueError("데이터가 로드되지 않았습니다. load_data()를 먼저 호출하세요.")
        return self.data.copy()

