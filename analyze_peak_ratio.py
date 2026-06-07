# =============================================================================
# 최고점 대비 매도비율의 합 기준 분석 스크립트
# =============================================================================
"""
이 스크립트는 "최고점 대비 매도비율의 합"이 가장 높은 파라미터를 찾습니다.

최고점 대비 매도비율의 합 = Σ(최고점대비매도가격비율 × 매도비중)
- 예: 82.20% 가격에서 7.08% 매도 → 82.20 × 7.08 / 100 = 5.82
- 모든 매도에 대해 합산하여 높을수록 좋은 지표입니다.
- 높은 값은 "최고점에 가까운 가격에서 많이 매도했다"는 의미입니다.
"""

import os
import sys
import glob
import pandas as pd
import multiprocessing
from datetime import datetime

def main():
    multiprocessing.freeze_support()
    
    print("=" * 80)
    print("🎯 최고점 대비 매도비율의 합 기준 분석")
    print("=" * 80)
    
    # 작업 디렉토리 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # SOXL 분석용 데이터 폴더
    data_dir = os.path.join(script_dir, "SOXL 분석용 데이터")
    
    # CSV 파일 찾기
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        print(f"❌ CSV 파일을 찾을 수 없습니다: {data_dir}")
        return
    
    print(f"\n📁 발견된 CSV 파일: {len(csv_files)}개")
    for csv_file in csv_files:
        print(f"  - {os.path.basename(csv_file)}")
    
    # 분석 설정
    max_combinations = 100000000  # 제한 없이 모든 조합 테스트
    n_cores = None
    chunk_size = 50
    
    all_results = []
    
    for file_idx, csv_path in enumerate(csv_files, 1):
        print(f"\n\n{'#'*80}")
        print(f"파일 {file_idx}/{len(csv_files)}: {os.path.basename(csv_path)}")
        print(f"{'#'*80}")
        
        result = analyze_single_file(csv_path, max_combinations, n_cores, chunk_size)
        if result:
            all_results.append(result)
    
    # 결과 저장
    save_peak_ratio_results(all_results)
    
    print("\n\n" + "=" * 80)
    print("🎉 분석 완료!")
    print("=" * 80)


def analyze_single_file(csv_path, max_combinations, n_cores, chunk_size):
    """단일 CSV 파일 분석"""
    from data_loader import DataLoader
    from technical_indicators import TechnicalIndicators
    from sell_signal_generator import SellSignalGenerator
    from optimizer import HyperparameterOptimizer
    import numpy as np
    
    print(f"\n📊 분석 시작: {os.path.basename(csv_path)}")
    
    try:
        # 데이터 로드
        data_loader = DataLoader(csv_path)
        data = data_loader.load_data()
        
        # 기술적 지표 계산
        indicators = TechnicalIndicators(data)
        data_with_indicators = indicators.get_data()
        
        # 매도 신호 생성기
        signal_generator = SellSignalGenerator(data_with_indicators)
        
        # 매수 평균단가 (저점의 2배)
        low_price = data['Close'].min()
        buy_avg_price = low_price * 2.0
        
        # 최고점
        peak_price = data['Close'].max()
        
        print(f"📈 최저가: ${low_price:.2f}")
        print(f"💰 매수 평균단가 (저점×2): ${buy_avg_price:.2f}")
        print(f"🔝 최고점: ${peak_price:.2f}")
        
        # 파일 타입 확인
        file_name = os.path.basename(csv_path)
        if 'SOXL' in file_name.upper():
            file_type = 'SOXL'
        elif 'USD' in file_name.upper():
            file_type = 'USD'
        else:
            file_type = 'SOXL'
        
        # 최적화 실행
        optimizer = HyperparameterOptimizer(
            data_with_indicators, 
            signal_generator, 
            buy_avg_price=buy_avg_price, 
            file_type=file_type
        )
        
        results = optimizer.grid_search(
            max_combinations=max_combinations,
            n_cores=n_cores,
            chunk_size=chunk_size
        )
        
        # 데이터 기간
        if isinstance(data.index, pd.DatetimeIndex):
            start_date = data.index.min()
            end_date = data.index.max()
            date_range = f"{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"
        else:
            date_range = "N/A"
        
        return {
            'file_name': file_name,
            'csv_path': csv_path,
            'optimizer': optimizer,
            'low_price': low_price,
            'buy_avg_price': buy_avg_price,
            'peak_price': peak_price,
            'date_range': date_range,
            'file_type': file_type
        }
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_peak_ratio_results(all_results):
    """최고점 대비 매도비율의 합 기준 결과 저장"""
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("\n\n" + "=" * 100)
    print("🏆 최고점 대비 매도비율의 합 기준 Top 10 전략")
    print("=" * 100)
    
    # 각 파일별 Top 10 출력 및 저장
    all_top10_data = []
    
    for result in all_results:
        optimizer = result['optimizer']
        file_name = result['file_name']
        
        # 최고점 대비 매도비율의 합 기준으로 Top 10 가져오기
        top_params = optimizer.get_best_parameters(
            top_n=10, 
            include_sell_records=True,
            sort_by='peak_ratio_weighted_sum'  # ← 핵심: 새로운 정렬 기준!
        )
        
        print(f"\n{'#'*100}")
        print(f"📁 {file_name} - 최고점 대비 매도비율의 합 기준 Top 10")
        print(f"   데이터 기간: {result['date_range']}")
        print(f"   최고점: ${result['peak_price']:.2f}")
        print(f"{'#'*100}")
        
        print(f"\n{'순위':<4} {'최고점대비합':<12} {'수익률':<10} {'매도비율':<10} {'가중수익률':<12} {'매도횟수':<8}")
        print(f"{'-'*70}")
        
        for rank, param_result in enumerate(top_params, 1):
            performance = param_result['performance']
            peak_ratio_sum = param_result.get('peak_ratio_weighted_sum', 0)
            weighted_return = param_result.get('weighted_return', 0)
            sell_ratio = param_result.get('sell_ratio', 0)
            
            print(f"{rank:<4} {peak_ratio_sum:<12.2f} {performance['total_return']:<10.2%} "
                  f"{sell_ratio:<10.2%} {weighted_return:<12.2%} {performance['num_trades']:<8}")
            
            # CSV 저장용 데이터
            params = param_result['params']
            all_top10_data.append({
                '파일명': file_name,
                '순위': rank,
                '최고점대비매도비율합': peak_ratio_sum,
                '수익률': performance['total_return'],
                '매도비율': sell_ratio,
                '가중수익률': weighted_return,
                '매도효율': performance['sell_efficiency'],
                '매도횟수': performance['num_trades'],
                'RSI_과매수': params.get('rsi_overbought'),
                'RSI_기간': params.get('rsi_period'),
                'RSI_가중치': params.get('rsi_weight'),
                'OBV_기간': params.get('obv_period'),
                'OBV_가중치': params.get('obv_weight'),
                'ATR_배수': params.get('atr_multiplier'),
                'ATR_가중치': params.get('atr_weight'),
                'BB_기간': params.get('bb_period'),
                'BB_시그마': params.get('bb_std_dev'),
                'BB_가중치': params.get('bb_weight'),
                'EMA_기간': params.get('ema_period'),
                '최대매도비율': params.get('max_sell_ratio'),
                '매도가중치베이스': params.get('sell_weight_base'),
                '매도가중치계수': params.get('sell_weight_coefficient'),
                '수익률가중치지수': params.get('price_weight_exponent'),
                '시간가중치_최대': params.get('time_weight_max'),
                '시간가중치_중간점': params.get('time_weight_midpoint'),
                '시간가중치_기울기': params.get('time_weight_slope')
            })
        
        # 1위 전략 상세 매도 기록 출력
        if top_params:
            print_sell_details(top_params[0], result['peak_price'])
    
    # CSV 저장
    df_results = pd.DataFrame(all_top10_data)
    csv_path = os.path.join('results', f'최고점대비매도비율합_Top10_{timestamp}.csv')
    df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 결과 저장: {csv_path}")


def print_sell_details(param_result, peak_price):
    """1위 전략의 상세 매도 기록 출력"""
    sell_records = param_result.get('sell_records', [])
    if not sell_records:
        return
    
    print(f"\n📋 1위 전략 상세 매도 기록 (총 {len(sell_records)}회)")
    print(f"{'='*100}")
    print(f"{'회차':<6} {'날짜':<12} {'매도가격':<10} {'매도비중':<10} {'최고점대비':<12} {'비율×비중':<12}")
    print(f"{'-'*100}")
    
    total_weighted = 0
    for record in sell_records:
        sell_count = record['sell_count']
        date = record['date']
        price = record['price']
        amount = record['amount']  # 매도 비중 (%)
        peak_ratio = record.get('peak_price_ratio', (price / peak_price) * 100)
        weighted = peak_ratio * amount / 100
        total_weighted += weighted
        
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
        
        print(f"{sell_count:<6} {date_str:<12} ${price:<9.2f} {amount:<9.2f}% {peak_ratio:<11.2f}% {weighted:<12.2f}")
    
    print(f"{'='*100}")
    print(f"📊 최고점 대비 매도비율의 합: {total_weighted:.2f}")
    print(f"   (해석: 높을수록 최고점에 가까운 가격에서 많이 매도함)")


if __name__ == '__main__':
    main()
