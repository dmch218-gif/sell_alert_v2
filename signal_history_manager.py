# =============================================================================
# 신호 이력 관리 모듈 (거래일 단위 카운트)
# =============================================================================
import json
import os
from datetime import datetime, date


class SignalHistoryManager:
    """
    신호 이력 관리 클래스
    
    - 거래일 단위로 신호 발생 횟수(sell_count) 관리
    - 같은 거래일 내 여러 신호는 1회로 카운트
    - 백테스팅과 동일한 방식으로 가중치 계산
    """
    
    def __init__(self, history_file: str = "signal_history.json"):
        """
        Args:
            history_file: 신호 이력 저장 파일 경로
        """
        self.history_file = history_file
        self.history = self._load_history()
    
    def _load_history(self) -> dict:
        """신호 이력 로드"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ 신호 이력 로드 실패: {e}")
                return {}
        return {}
    
    def _save_history(self):
        """신호 이력 저장"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ 신호 이력 저장 실패: {e}")
    
    def _get_trading_date(self) -> str:
        """
        현재 거래일 계산 (한국시간 기준)
        
        미국 장 시간을 고려하여 거래일 결정:
        - 한국시간 오전 8시 이전: 전일 거래일
        - 한국시간 오전 8시 이후: 당일 거래일
        
        Returns:
            거래일 문자열 (YYYY-MM-DD)
        """
        now = datetime.now()
        
        # 오전 8시 이전이면 전일 거래일로 간주 (미국 장 마감 전)
        if now.hour < 8:
            # 전일 날짜 계산
            from datetime import timedelta
            trading_date = (now - timedelta(days=1)).date()
        else:
            trading_date = now.date()
        
        return trading_date.strftime('%Y-%m-%d')
    
    def get_sell_count(self, symbol: str) -> int:
        """
        종목의 신호 발생 거래일 수 (sell_count) 반환
        
        Args:
            symbol: 종목 코드 (예: 'SOXL', 'USD')
        
        Returns:
            신호가 발생한 거래일 수
        """
        if symbol not in self.history:
            return 0
        
        signal_dates = self.history[symbol].get('signal_dates', [])
        return len(signal_dates)
    
    def has_signal_today(self, symbol: str) -> bool:
        """
        오늘(현재 거래일)에 이미 신호가 기록되었는지 확인
        
        Args:
            symbol: 종목 코드
        
        Returns:
            오늘 신호 기록 여부
        """
        if symbol not in self.history:
            return False
        
        trading_date = self._get_trading_date()
        signal_dates = self.history[symbol].get('signal_dates', [])
        
        return trading_date in signal_dates
    
    def record_signal(self, symbol: str, price: float = None, sell_ratio: float = None) -> bool:
        """
        신호 발생 기록 (거래일 단위)
        
        같은 거래일에 여러 번 호출되면:
        - sell_count는 1회만 카운트 (거래일 단위)
        - sell_ratio는 누적하여 합산
        - price는 가장 최근 가격으로 업데이트
        
        Args:
            symbol: 종목 코드
            price: 신호 발생 시 가격 (선택)
            sell_ratio: 매도 비율 (선택)
        
        Returns:
            새 거래일이면 True, 기존 거래일이면 False (하지만 비율은 누적됨)
        """
        trading_date = self._get_trading_date()
        
        # 종목 초기화
        if symbol not in self.history:
            self.history[symbol] = {
                'signal_dates': [],
                'details': {}
            }
        
        signal_dates = self.history[symbol]['signal_dates']
        is_new_trading_day = trading_date not in signal_dates
        
        # 새 거래일이면 날짜 추가
        if is_new_trading_day:
            signal_dates.append(trading_date)
        
        # 상세 정보 저장/업데이트
        if price is not None or sell_ratio is not None:
            if 'details' not in self.history[symbol]:
                self.history[symbol]['details'] = {}
            
            # 기존 데이터 가져오기 (누적용)
            existing = self.history[symbol]['details'].get(trading_date, {})
            existing_ratio = existing.get('sell_ratio', 0) or 0
            
            # 새 거래일이면 초기화, 아니면 누적
            if is_new_trading_day:
                new_ratio = sell_ratio or 0
                signal_count = 1
            else:
                new_ratio = existing_ratio + (sell_ratio or 0)
                signal_count = existing.get('signal_count', 1) + 1
            
            self.history[symbol]['details'][trading_date] = {
                'timestamp': datetime.now().isoformat(),
                'price': price,  # 가장 최근 가격
                'sell_ratio': new_ratio,  # 누적된 총 매도비율
                'signal_count': signal_count  # 해당 거래일의 신호 횟수
            }
        
        self._save_history()
        return is_new_trading_day
    
    def get_current_position(self, symbol: str) -> float:
        """
        현재 보유 비율 반환
        
        Args:
            symbol: 종목 코드
        
        Returns:
            현재 보유 비율 (0~100)
        """
        if symbol not in self.history:
            return 100.0
        
        return self.history[symbol].get('current_position', 100.0)
    
    def update_position(self, symbol: str, new_position: float):
        """
        보유 비율 업데이트
        
        Args:
            symbol: 종목 코드
            new_position: 새 보유 비율 (0~100)
        """
        if symbol not in self.history:
            self.history[symbol] = {
                'signal_dates': [],
                'details': {}
            }
        
        self.history[symbol]['current_position'] = max(0, min(100, new_position))
        self._save_history()
    
    def get_history_summary(self, symbol: str) -> dict:
        """
        종목의 신호 이력 요약 반환
        
        Args:
            symbol: 종목 코드
        
        Returns:
            이력 요약 딕셔너리
        """
        if symbol not in self.history:
            return {
                'sell_count': 0,
                'signal_dates': [],
                'current_position': 100.0,
                'has_signal_today': False,
                'details': {}
            }
        
        return {
            'sell_count': len(self.history[symbol].get('signal_dates', [])),
            'signal_dates': self.history[symbol].get('signal_dates', []),
            'current_position': self.history[symbol].get('current_position', 100.0),
            'has_signal_today': self.has_signal_today(symbol),
            'details': self.history[symbol].get('details', {})
        }
    
    def get_sell_history_table(self, symbol: str) -> list:
        """
        종목의 매도 이력을 테이블 형식으로 반환
        
        Args:
            symbol: 종목 코드
        
        Returns:
            매도 이력 리스트 [{date, price, sell_ratio, cumulative_ratio}, ...]
        """
        if symbol not in self.history:
            return []
        
        details = self.history[symbol].get('details', {})
        signal_dates = self.history[symbol].get('signal_dates', [])
        
        table = []
        cumulative_ratio = 0.0
        
        for i, date in enumerate(sorted(signal_dates)):
            detail = details.get(date, {})
            sell_ratio = detail.get('sell_ratio', 0) or 0
            price = detail.get('price', 0) or 0
            cumulative_ratio += sell_ratio
            
            table.append({
                'no': i + 1,
                'date': date,
                'price': price,
                'sell_ratio': sell_ratio,
                'cumulative_ratio': cumulative_ratio,
                'remaining_ratio': max(0, 1.0 - cumulative_ratio)
            })
        
        return table
    
    def reset_symbol(self, symbol: str):
        """
        종목의 신호 이력 초기화
        
        Args:
            symbol: 종목 코드
        """
        if symbol in self.history:
            del self.history[symbol]
            self._save_history()
    
    def reset_all(self):
        """모든 신호 이력 초기화"""
        self.history = {}
        self._save_history()


def test_signal_history():
    """신호 이력 관리 테스트"""
    print("=" * 60)
    print("신호 이력 관리 테스트")
    print("=" * 60)
    
    manager = SignalHistoryManager("test_signal_history.json")
    
    # 초기화
    manager.reset_all()
    
    # SOXL 테스트
    print("\n[SOXL 테스트]")
    print(f"초기 sell_count: {manager.get_sell_count('SOXL')}")
    
    # 첫 신호 기록
    result1 = manager.record_signal('SOXL', price=45.67, sell_ratio=0.0723)
    print(f"첫 신호 기록: {result1} (True 예상)")
    print(f"sell_count: {manager.get_sell_count('SOXL')}")
    
    # 같은 날 다시 기록 시도
    result2 = manager.record_signal('SOXL', price=46.00, sell_ratio=0.0550)
    print(f"같은 날 재기록: {result2} (False 예상)")
    print(f"sell_count: {manager.get_sell_count('SOXL')}")
    
    # 요약
    print(f"\n요약: {manager.get_history_summary('SOXL')}")
    
    # 테스트 파일 삭제
    import os
    if os.path.exists("test_signal_history.json"):
        os.remove("test_signal_history.json")
    
    print("\n✅ 테스트 완료")


if __name__ == "__main__":
    test_signal_history()

