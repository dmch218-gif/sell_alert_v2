# =============================================================================
# 텔레그램 알림 모듈
# =============================================================================
import requests
from datetime import datetime
import json
import os

class TelegramNotifier:
    """텔레그램 알림 클래스"""
    
    def __init__(self, bot_token: str, chat_id: str):
        """
        Args:
            bot_token: 텔레그램 봇 토큰
            chat_id: 채팅방 ID
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
        # 알림 기록 파일
        self.alert_log_file = "alert_history.json"
        self.alert_history = self._load_alert_history()
    
    def _load_alert_history(self) -> dict:
        """알림 기록 로드"""
        if os.path.exists(self.alert_log_file):
            try:
                with open(self.alert_log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_alert_history(self):
        """알림 기록 저장"""
        try:
            with open(self.alert_log_file, 'w', encoding='utf-8') as f:
                json.dump(self.alert_history, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"알림 기록 저장 실패: {e}")
    
    def _can_send_alert(self, symbol: str, cooldown_hours: int = 24) -> bool:
        """
        쿨다운 체크 (같은 종목에 대해 연속 알림 방지)
        
        Args:
            symbol: 종목 코드
            cooldown_hours: 쿨다운 시간 (시간 단위)
        
        Returns:
            알림 전송 가능 여부
        """
        if symbol not in self.alert_history:
            return True
        
        last_alert_time = datetime.fromisoformat(self.alert_history[symbol]['timestamp'])
        elapsed_hours = (datetime.now() - last_alert_time).total_seconds() / 3600
        
        return elapsed_hours >= cooldown_hours
    
    def _record_alert(self, symbol: str, signal_info: dict):
        """알림 기록"""
        self.alert_history[symbol] = {
            'timestamp': datetime.now().isoformat(),
            'signal_strength': signal_info.get('signal_strength', 0),
            'signal_type': signal_info.get('signal_type', 'N/A'),
            'price': signal_info.get('price', 0)
        }
        self._save_alert_history()
    
    def send_message(self, message: str) -> bool:
        """
        텔레그램 메시지 전송
        
        Args:
            message: 전송할 메시지
            
        Returns:
            전송 성공 여부
        """
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, data=data, timeout=10)
            result = response.json()
            
            if result.get('ok'):
                print(f"✅ 텔레그램 메시지 전송 성공")
                return True
            else:
                print(f"❌ 텔레그램 전송 실패: {result.get('description')}")
                return False
                
        except requests.exceptions.Timeout:
            print("❌ 텔레그램 전송 타임아웃")
            return False
        except Exception as e:
            print(f"❌ 텔레그램 전송 오류: {e}")
            return False
    
    def send_sell_signal_alert(self, symbol: str, symbol_name: str, signal_info: dict, cooldown_hours: int = 24) -> bool:
        """
        매도 신호 알림 전송
        
        Args:
            symbol: 종목 코드
            symbol_name: 종목 이름
            signal_info: 신호 정보 딕셔너리
            cooldown_hours: 쿨다운 시간
            
        Returns:
            전송 성공 여부
        """
        # 쿨다운 체크
        if not self._can_send_alert(symbol, cooldown_hours):
            print(f"⏸️ {symbol}: 쿨다운 중 (마지막 알림 후 {cooldown_hours}시간 미경과)")
            return False
        
        # 메시지 구성
        signal_indicators = '+'.join(signal_info.get('signal_indicators', [])) or 'N/A'
        
        message = f"""
🚨 <b>매도 신호 발생!</b>

📊 <b>종목:</b> {symbol_name}
📅 <b>날짜:</b> {signal_info['date'].strftime('%Y-%m-%d')}

💰 <b>현재가:</b> ${signal_info['price']:,.2f}
⏱️ <b>보유일수:</b> {signal_info['days_held']}일 (저점 대비: {signal_info['days_from_low']}일)

🎯 <b>신호 강도:</b> {signal_info['signal_strength']:.4f} (최대: {signal_info['max_possible_score']:.0f})
📍 <b>신호 지표:</b> {signal_indicators}
📐 <b>정규화값:</b> {signal_info['normalized_score']:.4f}

<b>[가중치]</b>
├ 매도가중치: {signal_info['sell_weight']:.4f}
├ 수익률가중치: {signal_info['price_weight']:.4f}
└ 시간가중치: {signal_info['time_weight']:.4f}

📉 <b>전체기준 매도비율:</b> {signal_info['total_sell_ratio']*100:.2f}%
📉 <b>보유기준 매도비율:</b> {signal_info['hold_based_sell_ratio']*100:.2f}%

⚠️ 위 정보는 참고용이며, 투자 결정은 본인 판단에 따라 해주세요.
        """.strip()
        
        success = self.send_message(message)
        
        if success:
            self._record_alert(symbol, signal_info)
        
        return success
    
    def send_daily_summary(self, summaries: list, signal_history: dict = None, all_sell_history: dict = None) -> bool:
        """
        일일 요약 알림 전송 (장 마감 후)
        
        종가 기준으로 분석한 매도 비율을 메인으로 표시
        실시간 신호 이력은 참고 섹션으로 분리
        
        Args:
            summaries: 종목별 요약 리스트 (종가 기준)
            signal_history: 금일 실시간 신호 이력 딕셔너리 {symbol: [signal_list]}
            all_sell_history: 전체 매도 이력 딕셔너리 {symbol: [history_table]}
            
        Returns:
            전송 성공 여부
        """
        now = datetime.now()
        
        message = f"""
📊 <b>일일 장 마감 보고 (종가 기준)</b>
📅 {now.strftime('%Y-%m-%d %H:%M')}

{'─' * 20}
"""
        for summary in summaries:
            signal_status = "🔴 매도 신호" if summary.get('has_signal') else "🟢 신호 없음"
            price_change = summary.get('change_percent', 0)
            change_emoji = "📈" if price_change >= 0 else "📉"
            
            # 종가 기준 매도비율 (summaries에서 가져옴)
            closing_sell_ratio = summary.get('total_sell_ratio', 0)
            
            # 신호 지표 표시 (신호 강도가 0보다 크면)
            signal_indicators = summary.get('signal_indicators', [])
            indicators_str = '+'.join(signal_indicators) if signal_indicators else 'N/A'
            
            message += f"""
<b>{summary['name']}</b>
├ 종가: ${summary['price']:,.2f} {change_emoji} {price_change:+.2f}%
├ 보유일수: {summary.get('days_held', 'N/A')}일
├ 신호 강도: {summary['signal_strength']:.4f}"""
            
            # 신호 강도가 0보다 크면 신호 지표 표시
            if summary['signal_strength'] > 0:
                message += f"""
├ 신호 지표: {indicators_str}"""
            
            message += f"""
└ 상태: {signal_status}
"""
            # =====================================================
            # 📍 종가 기준 매도비율 (메인)
            # =====================================================
            if summary.get('has_signal') and closing_sell_ratio > 0:
                message += f"""
🎯 <b>종가 기준 매도비율: {closing_sell_ratio*100:.2f}%</b>
   (전날 보유비중 기준, 종가로 계산)

💡 <b>장 마감 후 매도 시:</b>
   → 전날 보유비중의 <b>{closing_sell_ratio*100:.2f}%</b>를 매도하세요.
"""
            else:
                message += "\n📋 종가 기준 매도 신호 없음\n"
            
            # =====================================================
            # 📍 장중 실시간 신호 이력 (참고용)
            # =====================================================
            signal_count = 0
            total_realtime_ratio = 0
            if signal_history and summary.get('symbol') in signal_history:
                signals = signal_history[summary['symbol']]
                if signals:
                    total_realtime_ratio = sum(sig['ratio'] for sig in signals)
                    signal_count = len(signals)
            
            if signal_count > 0:
                message += f"""
<b>📋 [참고] 장중 실시간 신호 ({signal_count}건)</b>
"""
                for i, sig in enumerate(signals, 1):
                    indicators = '+'.join(sig.get('signal_indicators', [])) or 'N/A'
                    message += f"  #{i} {sig['time']} | ${sig['price']:,.2f} | {sig.get('ratio', 0)*100:.2f}% | {indicators}\n"
                
                message += f"  (실시간 합계: {total_realtime_ratio*100:.2f}% - 참고용)\n"
            
            # =====================================================
            # 📍 전체 매도 이력 테이블
            # =====================================================
            symbol = summary.get('symbol')
            if all_sell_history and symbol in all_sell_history:
                history = all_sell_history[symbol]
                if history and len(history) > 0:
                    message += f"""
<b>📈 전체 매도 이력 ({len(history)}회 거래일)</b>
<code>No  날짜        가격     매도%   누적%   잔여%</code>
"""
                    for h in history:
                        message += f"<code>{h['no']:2d}  {h['date'][5:]}  ${h['price']:7.2f}  {h['sell_ratio']*100:5.2f}%  {h['cumulative_ratio']*100:5.2f}%  {h['remaining_ratio']*100:5.2f}%</code>\n"
                    
                    # 총 누적 매도 비율
                    total_cumulative = history[-1]['cumulative_ratio']
                    remaining = history[-1]['remaining_ratio']
                    message += f"\n📊 <b>총 누적매도: {total_cumulative*100:.2f}% | 잔여보유: {remaining*100:.2f}%</b>\n"
            
            message += "\n"
        
        message += f"""{'─' * 20}
📌 <b>매도는 '종가 기준 매도비율'을 참고하세요.</b>
⚠️ 본 보고는 참고용이며, 투자 결정은 본인 판단에 따라 해주세요."""
        
        return self.send_message(message.strip())
    
    def send_realtime_signal_alert(self, symbol: str, symbol_name: str, signal_info: dict, current_price: float, 
                                     cumulative_sell_ratio: float = 0.0, sell_history: list = None) -> bool:
        """
        실시간 매도 신호 알림 전송 (장중)
        
        Args:
            symbol: 종목 코드
            symbol_name: 종목 이름
            signal_info: 신호 정보 딕셔너리
            current_price: 현재가 (실시간)
            cumulative_sell_ratio: 금일 누적 매도비율 (이번 매도 포함)
            sell_history: 전체 매도 이력 테이블 (선택)
            
        Returns:
            전송 성공 여부
        """
        now = datetime.now()
        signal_indicators = '+'.join(signal_info.get('signal_indicators', [])) or 'N/A'
        
        # 매도 후 전체 보유비율 계산 (100% - 누적 매도비율)
        remaining_position = max(0, 100 - cumulative_sell_ratio * 100)
        
        message = f"""
🚨 <b>매도 신호 발생!</b>

📊 <b>종목:</b> {symbol_name}
⏰ <b>시간:</b> {now.strftime('%Y-%m-%d %H:%M:%S')}

<b>━━━ 매도 정보 ━━━</b>
💰 매도가격: <b>${current_price:,.2f}</b>
📉 전체기준 매도비율: <b>{signal_info['total_sell_ratio']*100:.2f}%</b>
📊 보유기준 매도비율: <b>{signal_info['hold_based_sell_ratio']*100:.2f}%</b>

<b>━━━ 신호 정보 ━━━</b>
📍 신호지표: <b>{signal_indicators}</b>
🎯 신호강도: {signal_info['signal_strength']:.4f}
📐 정규화값: {signal_info['normalized_score']:.4f}

<b>━━━ 가중치 ━━━</b>
⚖️ 신호가중치: {signal_info['sell_weight']:.4f}
💹 수익률가중치: {signal_info['price_weight']:.4f}
⏳ 시간가중치: {signal_info['time_weight']:.4f}

<b>━━━ 보유 현황 ━━━</b>
⏱️ 보유일수: {signal_info['days_held']}일
📦 매도후 전체 보유비율: <b>{remaining_position:.2f}%</b>
"""
        
        # 매도 이력 테이블 추가
        if sell_history and len(sell_history) > 0:
            message += f"""
<b>━━━ 전체 매도 이력 ({len(sell_history)}건) ━━━</b>
<code>No  날짜        가격     매도%   누적%</code>
"""
            for h in sell_history[-5:]:  # 최근 5건만 표시
                message += f"<code>{h['no']:2d}  {h['date'][5:]}  ${h['price']:7.2f}  {h['sell_ratio']*100:5.2f}%  {h['cumulative_ratio']*100:5.2f}%</code>\n"
            
            if len(sell_history) > 5:
                message += f"<code>... 외 {len(sell_history) - 5}건 (총 {len(sell_history)}회 거래일)</code>\n"
            
            # 총 누적 매도 비율
            total_cumulative = sell_history[-1]['cumulative_ratio'] if sell_history else 0
            message += f"\n📊 <b>전체 누적 매도: {total_cumulative*100:.2f}%</b>"

        message += """

📌 <b>장중 실시간 신호 (참고용)</b>
   → 최종 매도는 장 마감 후 '종가 기준' 일일보고를 참고하세요.
⚠️ 투자 결정은 본인 판단에 따라 해주세요."""
        
        success = self.send_message(message.strip())
        
        if success:
            self._record_alert(symbol, signal_info)
        
        return success
    
    def send_startup_message(self) -> bool:
        """시작 알림 전송"""
        message = f"""
🚀 <b>매도 신호 알림 시스템 시작</b>

📅 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
✅ 시스템이 정상적으로 시작되었습니다.

모니터링 중인 종목:
• SOXL (반도체 3배 레버리지)
• USD (반도체 2배 레버리지)
        """.strip()
        
        return self.send_message(message)
    
    def test_connection(self) -> bool:
        """텔레그램 연결 테스트"""
        try:
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=10)
            result = response.json()
            
            if result.get('ok'):
                bot_info = result.get('result', {})
                print(f"✅ 텔레그램 봇 연결 성공: @{bot_info.get('username')}")
                return True
            else:
                print(f"❌ 텔레그램 봇 연결 실패: {result.get('description')}")
                return False
                
        except Exception as e:
            print(f"❌ 텔레그램 연결 오류: {e}")
            return False


def test_notifier():
    """알림 테스트"""
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    
    print("=" * 60)
    print("텔레그램 알림 테스트")
    print("=" * 60)
    
    if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ config.py에서 TELEGRAM_BOT_TOKEN을 설정해주세요.")
        print("\n텔레그램 봇 생성 방법:")
        print("1. 텔레그램에서 @BotFather 검색")
        print("2. /newbot 명령 입력")
        print("3. 봇 이름 입력")
        print("4. 발급된 토큰을 config.py에 입력")
        print("\n채팅 ID 확인 방법:")
        print("1. 생성한 봇과 대화 시작 (/start)")
        print("2. @userinfobot 에게 메시지 전송")
        print("3. 표시되는 ID를 config.py에 입력")
        return
    
    notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    
    # 연결 테스트
    if notifier.test_connection():
        # 테스트 메시지 전송
        test_message = f"🧪 테스트 메시지\n\n현재 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n알림 시스템이 정상 작동합니다!"
        notifier.send_message(test_message)


if __name__ == "__main__":
    test_notifier()

