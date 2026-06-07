# =============================================================================
# SOXL 매도 방법론 백테스트 시스템 - 메인 실행 파일
# =============================================================================
import os
import sys
import multiprocessing
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime

def calculate_buy_avg_price_from_low(data):
    """
    저점 대비 2배 가격으로 매수 평균단가 계산
    
    Args:
        data: 데이터프레임 (Close 컬럼 포함)
    
    Returns:
        매수 평균단가 (저점의 2배)
    """
    low_price = data['Close'].min()  # 전체 기간 최저가
    buy_avg_price = low_price * 2.0  # 저점의 2배
    return buy_avg_price

def create_sell_price_histogram(df_sells, file_name, csv_path=None, total_return=None, save_dir=None, show=False):
    """
    가격대별 매도 비중 및 매도 금액 히스토그램 생성 (이중 Y축)
    첫 번째 열에 매도횟수별 분석 그래프 포함
    
    Args:
        df_sells: 매도 기록 데이터프레임
        file_name: 파일명
        csv_path: 원본 CSV 파일 경로 (선택사항)
        total_return: 총 수익률 (선택사항)
        save_dir: 저장 디렉토리 (None이면 표시만)
        show: 그래프 표시 여부 (기본 False)
    """
    if df_sells.empty:
        return
    
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
    plt.rcParams['axes.unicode_minus'] = False
    
    # 매도 횟수별로 정렬
    df_sells = df_sells.sort_values('sell_count')
    
    # 가격대별 매도 비중 및 금액 계산
    prices = df_sells['price'].values
    amounts = df_sells['amount'].values  # 매도 비중 (%)
    sell_counts = df_sells['sell_count'].values
    # 매도 금액 계산 (비중 * 가격, 상대적 금액)
    sell_amounts = amounts * prices  # 매도 금액 (비중% * 가격)
    
    # 히스토그램 구간 설정
    min_price = prices.min()
    max_price = prices.max()
    num_bins = min(20, len(df_sells))  # 최대 20개 구간
    
    # 가격 구간별 매도 비중 및 금액 합계 계산
    bins = np.linspace(min_price, max_price, num_bins + 1)
    bin_indices = np.digitize(prices, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
    
    # 각 구간별 매도 비중 및 금액 합계
    bin_amounts = np.zeros(len(bins) - 1)
    bin_sell_amounts = np.zeros(len(bins) - 1)
    for i, (amount, sell_amount) in enumerate(zip(amounts, sell_amounts)):
        bin_amounts[bin_indices[i]] += amount
        bin_sell_amounts[bin_indices[i]] += sell_amount
    
    # 그래프 생성 (2개 subplot: 첫 번째는 매도횟수별, 두 번째는 가격대별)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 7))
    
    # 첫 번째 subplot: 매도 횟수별 매도 금액 막대그래프
    bars0 = ax0.bar(sell_counts, sell_amounts, color='steelblue', alpha=0.7, 
                   edgecolor='black', linewidth=0.5)
    ax0.set_xlabel('매도 횟수', fontsize=12, fontweight='bold')
    ax0.set_ylabel('매도 금액 (비중% × 가격)', fontsize=12, fontweight='bold')
    ax0.set_title(f'{file_name} - 매도 횟수별 매도 금액', fontsize=14, fontweight='bold')
    ax0.grid(True, alpha=0.3, axis='y')
    ax0.set_xticks(sell_counts)
    
    # 각 막대 위에 값 표시
    for count, amount in zip(sell_counts, sell_amounts):
        ax0.text(count, amount, f'{amount:.1f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 두 번째 subplot: 가격대별 히스토그램 (이중 Y축)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0]) * 0.6
    
    # 첫 번째 Y축: 매도 비중 (%)
    bars1 = ax1.bar(bin_centers - width/2, bin_amounts, width=width, 
                    color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5,
                    label='매도 비중 (%)')
    ax1.set_xlabel('매도 가격 ($)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('매도 비중 (%)', fontsize=12, fontweight='bold', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 각 막대 위에 비중 값 표시
    for center, amount in zip(bin_centers, bin_amounts):
        if amount > 0:
            ax1.text(center - width/2, amount, f'{amount:.1f}%', 
                    ha='center', va='bottom', fontsize=7, fontweight='bold', color='steelblue')
    
    # 두 번째 Y축: 매도 금액
    ax2 = ax1.twinx()
    bars2 = ax2.bar(bin_centers + width/2, bin_sell_amounts, width=width,
                    color='coral', alpha=0.7, edgecolor='black', linewidth=0.5,
                    label='매도 금액 (비중% × 가격)')
    ax2.set_ylabel('매도 금액', fontsize=12, fontweight='bold', color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')
    
    # 각 막대 위에 금액 값 표시
    for center, amount in zip(bin_centers, bin_sell_amounts):
        if amount > 0:
            ax2.text(center + width/2, amount, f'{amount:.1f}', 
                    ha='center', va='bottom', fontsize=7, fontweight='bold', color='coral')
    
    # 범례
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax1.set_title(f'{file_name} - 가격대별 매도 비중 및 매도 금액 분포', fontsize=14, fontweight='bold')
    
    # 통계 정보 계산
    total_sold = bin_amounts.sum()  # 총 매도 비중 (%)
    total_sell_amount = bin_sell_amounts.sum()  # 총 매도 금액
    avg_price = (prices * amounts).sum() / amounts.sum() if amounts.sum() > 0 else 0  # 가중 평균 매도 가격
    
    # 평균 매도 금액 = 총 매도 금액 / 매도한 총 주식수 (amounts의 합)
    total_shares_sold = amounts.sum()  # 매도한 총 주식수 (비중 %)
    avg_sell_amount = total_sell_amount / total_shares_sold if total_shares_sold > 0 else 0
    
    # 통계 정보 텍스트
    stats_text = f'총 매도 비중: {total_sold:.2f}%\n총 매도 금액: {total_sell_amount:.2f}\n평균 매도 가격: ${avg_price:.2f}\n평균 매도 금액: ${avg_sell_amount:.2f}'
    if total_return is not None:
        stats_text += f'\n수익률: {total_return:.2%}'
    
    # 통계 정보를 첫 번째 subplot에 표시
    ax0.text(0.02, 0.98, stats_text, transform=ax0.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 저장 (항상 저장)
    if save_dir is None:
        save_dir = 'graphs'  # 기본 저장 디렉토리
    os.makedirs(save_dir, exist_ok=True)
    safe_filename = file_name.replace('.csv', '').replace('.CSV', '')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f'{safe_filename}_가격대별_매도분포_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  💾 히스토그램 저장: {save_path}")
    
    # 표시 여부에 따라 결정
    if show:
        plt.show()
    else:
        plt.close()

def create_sell_count_charts(df_sells, file_name, csv_path=None, date_range=None, total_return=None, save_dir=None, show=False, criteria=None, performance=None):
    """
    매도 분석 그래프 생성 (4개 subplot)
    1. 가격대별 매도 비중 및 매도금액 분포
    2. 매도 횟수별 매도 금액
    3. 매도 횟수별 매도 가격 추이
    4. 날짜별 가격 추이 및 매도 시점
    
    Args:
        df_sells: 매도 기록 데이터프레임 (date, price, amount, sell_count 포함)
        file_name: 파일명
        csv_path: 원본 CSV 파일 경로 (날짜별 가격 그래프용, 선택사항)
        date_range: 데이터 기간 (예: "2020-01-01 ~ 2023-12-31")
        total_return: 총 수익률 (선택사항)
        save_dir: 저장 디렉토리 (None이면 표시만)
        show: 그래프 표시 여부 (기본 False)
        criteria: 분석 기준 (예: "수익률 합계 기준", "수익률-편차 기준", "개별 수익률 기준")
        performance: 백테스트 결과의 performance 딕셔너리 (avg_price, total_return 등 포함, 선택사항)
    """
    if df_sells.empty:
        return
    
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
    plt.rcParams['axes.unicode_minus'] = False
    
    # 매도 횟수별로 정렬
    df_sells = df_sells.sort_values('sell_count')
    
    sell_counts = df_sells['sell_count'].values
    prices = df_sells['price'].values
    amounts = df_sells['amount'].values  # 매도 비중 (%)
    # 매도 금액 계산 (비중 * 가격)
    sell_amounts = amounts * prices
    
    # 날짜 정보 확인
    has_date = 'date' in df_sells.columns
    
    # 원본 CSV 파일 로드 (최고점 계산 및 날짜별 가격 그래프용)
    original_data = None
    peak_price = None
    if csv_path is not None:
        try:
            from data_loader import DataLoader
            data_loader = DataLoader(csv_path)
            original_data = data_loader.load_data()
            if 'Close' in original_data.columns:
                peak_price = original_data['Close'].max()
        except Exception as e:
            print(f"  ⚠️ 원본 CSV 파일 로드 실패: {e}")
            original_data = None
    
    # 가격대별 매도분포를 위한 히스토그램 구간 설정
    min_price = prices.min()
    max_price = prices.max()
    num_bins = min(20, len(df_sells))  # 최대 20개 구간
    bins = np.linspace(min_price, max_price, num_bins + 1)
    bin_indices = np.digitize(prices, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
    
    # 각 구간별 매도 비중 및 금액 합계
    bin_amounts = np.zeros(len(bins) - 1)
    bin_sell_amounts = np.zeros(len(bins) - 1)
    for i, (amount, sell_amount) in enumerate(zip(amounts, sell_amounts)):
        bin_amounts[bin_indices[i]] += amount
        bin_sell_amounts[bin_indices[i]] += sell_amount
    
    # 그래프 생성 (4개 subplot)
    if original_data is not None and has_date:
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(16, 16))
    else:
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(16, 12))
    
    # 첫 번째 그래프: 가격대별 매도 비중 및 매도금액 분포 (이중 Y축)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0]) * 0.6
    
    # 첫 번째 Y축: 매도 비중 (%)
    bars0_1 = ax0.bar(bin_centers - width/2, bin_amounts, width=width, 
                      color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax0.set_xlabel('매도 가격 ($)', fontsize=12, fontweight='bold')
    ax0.set_ylabel('매도 비중 (%)', fontsize=12, fontweight='bold', color='steelblue')
    ax0.tick_params(axis='y', labelcolor='steelblue')
    ax0.grid(True, alpha=0.3, axis='y')
    
    # 각 막대 위에 비중 값 표시
    for center, amount in zip(bin_centers, bin_amounts):
        if amount > 0:
            ax0.text(center - width/2, amount, f'{amount:.1f}%', 
                    ha='center', va='bottom', fontsize=7, fontweight='bold', color='steelblue')
    
    # 두 번째 Y축: 매도 금액
    ax0_2 = ax0.twinx()
    bars0_2 = ax0_2.bar(bin_centers + width/2, bin_sell_amounts, width=width,
                        color='coral', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax0_2.set_ylabel('매도 금액', fontsize=12, fontweight='bold', color='coral')
    ax0_2.tick_params(axis='y', labelcolor='coral')
    
    # 각 막대 위에 금액 값 표시
    for center, amount in zip(bin_centers, bin_sell_amounts):
        if amount > 0:
            ax0_2.text(center + width/2, amount, f'{amount:.1f}', 
                      ha='center', va='bottom', fontsize=7, fontweight='bold', color='coral')
    
    # 범례 삭제 - 제목만 설정
    ax0.set_title(f'{file_name} - 가격대별 매도 비중 및 매도금액 분포', fontsize=14, fontweight='bold')
    
    # 두 번째 그래프: 매도 횟수별 매도 금액 막대그래프
    bars1 = ax1.bar(sell_counts, sell_amounts, color='steelblue', alpha=0.7, 
                   edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('매도 횟수', fontsize=12, fontweight='bold')
    ax1.set_ylabel('매도 금액 (비중% × 가격)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{file_name} - 매도 횟수별 매도 금액', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(sell_counts)
    
    # 각 막대 위에 값 표시
    for count, amount in zip(sell_counts, sell_amounts):
        ax1.text(count, amount, f'{amount:.1f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 세 번째 그래프: 매도 가격 꺾은선 그래프
    ax2.plot(sell_counts, prices, marker='o', linewidth=2, markersize=8, 
             color='coral', markerfacecolor='red', markeredgecolor='darkred')
    ax2.set_xlabel('매도 횟수', fontsize=12, fontweight='bold')
    ax2.set_ylabel('매도 가격 ($)', fontsize=12, fontweight='bold')
    ax2.set_title(f'{file_name} - 매도 횟수별 매도 가격 추이', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(sell_counts)
    
    # 각 점에 가격 표시
    for count, price in zip(sell_counts, prices):
        ax2.text(count, price, f'${price:.2f}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 네 번째 그래프: 날짜별 가격 그래프에 매도 시점 표시
    if original_data is not None and has_date:
        # 원본 데이터의 날짜와 가격 (인덱스가 날짜)
        if isinstance(original_data.index, pd.DatetimeIndex):
            dates = original_data.index
        elif 'Date' in original_data.columns:
            dates = pd.to_datetime(original_data['Date'])
        else:
            dates = pd.date_range(start='2020-01-01', periods=len(original_data), freq='D')
        
        close_prices = original_data['Close'].values
        
        # 최고점과 최저점 찾기
        max_idx = np.argmax(close_prices)
        min_idx = np.argmin(close_prices)
        max_price = close_prices[max_idx]
        min_price = close_prices[min_idx]
        max_date = dates[max_idx]
        min_date = dates[min_idx]
        
        # 가격 그래프
        ax3.plot(dates, close_prices, linewidth=1.5, color='gray', alpha=0.7, label='종가')
        
        # 최고점 표시
        ax3.scatter([max_date], [max_price], s=200, color='green', marker='^', 
                   edgecolors='darkgreen', linewidths=2, zorder=6, 
                   label=f'최고점: ${max_price:.2f}', alpha=0.9)
        
        # 최저점 표시
        ax3.scatter([min_date], [min_price], s=200, color='blue', marker='v', 
                   edgecolors='darkblue', linewidths=2, zorder=6, 
                   label=f'최저점: ${min_price:.2f}', alpha=0.9)
        
        ax3.set_xlabel('날짜', fontsize=12, fontweight='bold')
        ax3.set_ylabel('가격 ($)', fontsize=12, fontweight='bold')
        ax3.set_title(f'{file_name} - 날짜별 가격 추이 및 매도 시점', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 매도 시점에 포인트 표시 (5% 초과 매도만 표시)
        df_sells_filtered = df_sells[df_sells['amount'] > 5]
        sell_dates = pd.to_datetime(df_sells_filtered['date'].values)
        sell_prices = df_sells_filtered['price'].values
        
        # 매도 포인트 표시 (투명도 적용)
        ax3.scatter(sell_dates, sell_prices, s=150, color='red', marker='v', 
                   edgecolors='darkred', linewidths=2, zorder=5, label='매도 시점', alpha=0.7)
        
        # 각 포인트에 가격만 표시
        for date, price in zip(sell_dates, sell_prices):
            text = f'${price:.2f}'
            ax3.annotate(text, xy=(date, price), xytext=(10, 10), 
                        textcoords='offset points', fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red', lw=1))
        
        # 날짜 형식 설정
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 범례 표시
        ax3.legend(loc='best', fontsize=9)
    
    # 통계 정보 계산
    total_sold = amounts.sum()  # 총 매도 비중 (%)
    total_amount = sell_amounts.sum()  # 총 매도 금액
    
    # performance 딕셔너리가 제공된 경우 해당 값 사용, 아니면 df_sells에서 계산
    if performance is not None:
        avg_price = performance.get('avg_price', 0)
        total_return = performance.get('total_return', total_return)
        total_sold = performance.get('total_sold', total_sold)
    else:
        avg_price = (prices * amounts).sum() / amounts.sum() if amounts.sum() > 0 else 0  # 가중 평균 매도 가격
    
    # 최고점 대비 가중평균매도금액 비율 계산
    peak_price_ratio = None
    if peak_price is not None and peak_price > 0:
        peak_price_ratio = (avg_price / peak_price) * 100
    
    # 통계 정보 텍스트 (기간, 총 매도 비중, 가중평균 매도 가격, 최고점 대비 비율, 수익률)
    stats_text = f'기간: {date_range if date_range else "N/A"}\n총 매도 비중: {total_sold:.2f}%\n가중평균 매도 가격: ${avg_price:.2f}'
    if peak_price_ratio is not None:
        stats_text += f'\n최고점 대비 매도가격: {peak_price_ratio:.2f}%'
    if total_return is not None:
        stats_text += f'\n수익률: {total_return:.2%}'
    
    # 통계 정보를 첫 번째 subplot에 표시
    ax0.text(0.02, 0.98, stats_text, transform=ax0.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 저장 (항상 저장)
    if save_dir is None:
        save_dir = 'graphs'  # 기본 저장 디렉토리
    os.makedirs(save_dir, exist_ok=True)
    
    # 파일명 생성: USD_기준_기간_매도 분석 그래프_날짜_시간
    safe_filename = file_name.replace('.csv', '').replace('.CSV', '')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 기준 문자열 추가
    criteria_str = f"_{criteria}" if criteria else ""
    
    # 기간 정보 추출 (date_range에서 날짜 부분만 추출)
    if date_range and date_range != 'N/A':
        period_str = date_range.replace(' ~ ', '_').replace('-', '').replace(' ', '')
        save_path = os.path.join(save_dir, f'{safe_filename}{criteria_str}_{period_str}_매도 분석 그래프_{timestamp}.png')
    else:
        # 기간 정보가 없을 때는 기간 부분 제외
        save_path = os.path.join(save_dir, f'{safe_filename}{criteria_str}_매도 분석 그래프_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  💾 그래프 저장: {save_path}")
    
    # 표시 여부에 따라 결정
    if show:
        plt.show()
    else:
        plt.close()

def analyze_single_file(csv_path, max_combinations=100000, n_cores=None, chunk_size=50):
    """
    단일 CSV 파일에 대한 분석 실행
    
    Args:
        csv_path: CSV 파일 경로
        max_combinations: 최대 조합 수
        n_cores: CPU 코어 수
        chunk_size: 청크 크기
    
    Returns:
        분석 결과 딕셔너리
    """
    print(f"\n{'='*80}")
    print(f"📊 분석 시작: {os.path.basename(csv_path)}")
    print(f"{'='*80}")
    
    try:
        # 1. 데이터 로드
        from data_loader import DataLoader
        
        data_loader = DataLoader(csv_path)
        data = data_loader.load_data()
        
        # 2. 기술적 지표 계산
        from technical_indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators(data)
        data_with_indicators = indicators.get_data()
        
        # 3. 매도 신호 생성기 초기화
        from sell_signal_generator import SellSignalGenerator
        
        signal_generator = SellSignalGenerator(data_with_indicators)
        
        # 4. 매수 평균단가 계산 (저점 대비 2배)
        low_price = data['Close'].min()
        buy_avg_price = calculate_buy_avg_price_from_low(data)
        print(f"📈 최저가: ${low_price:.2f}")
        print(f"💰 매수 평균단가 (저점×2): ${buy_avg_price:.2f}")
        
        # 데이터 기간 정보 추출 (Date가 인덱스로 설정되어 있을 수 있음)
        if isinstance(data.index, pd.DatetimeIndex):
            start_date = data.index.min()
            end_date = data.index.max()
            date_range = f"{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"
        elif 'Date' in data.columns:
            start_date = data['Date'].min()
            end_date = data['Date'].max()
            date_range = f"{start_date} ~ {end_date}"
        else:
            date_range = "날짜 정보 없음"
        
        # 5. 하이퍼파라미터 최적화
        from optimizer import HyperparameterOptimizer
        
        # 파일명에서 파일 타입 확인 (SOXL 또는 USD)
        file_name = os.path.basename(csv_path)
        if 'SOXL' in file_name.upper():
            file_type = 'SOXL'
        elif 'USD' in file_name.upper():
            file_type = 'USD'
        else:
            file_type = 'SOXL'  # 기본값은 SOXL
        
        optimizer = HyperparameterOptimizer(data_with_indicators, signal_generator, buy_avg_price=buy_avg_price, file_type=file_type)
        
        # 최적화 실행
        results = optimizer.grid_search(
            max_combinations=max_combinations,
            n_cores=n_cores,
            chunk_size=chunk_size
        )
        
        return {
            'file_name': os.path.basename(csv_path),
            'low_price': low_price,
            'buy_avg_price': buy_avg_price,
            'optimizer': optimizer,
            'results': results,
            'data': data_with_indicators,  # 기술적 지표가 포함된 데이터
            'csv_path': csv_path,  # 원본 CSV 파일 경로 (그래프용)
            'date_range': date_range,  # 데이터 기간
            'file_type': file_type  # 파일 타입
        }
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

# Windows에서 멀티프로세싱 지원
if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    # 콘솔 출력 설정 확인 (디버깅용)
    print("=" * 80)
    print("🎯 매도 방법론 백테스트 시스템 (직박구리 폴더)")
    print("=" * 80)
    print(f"\n📋 콘솔 출력 설정:")
    print(f"   - sys.stdout: {sys.stdout}")
    print(f"   - sys.stdout.encoding: {sys.stdout.encoding}")
    print(f"   - sys.stdout.isatty(): {sys.stdout.isatty()}")
    print(f"   - PYTHONIOENCODING: {os.environ.get('PYTHONIOENCODING', 'Not set')}")
    print("=" * 80)
    
    # 작업 디렉토리 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 직박구리 폴더 경로
    jikbakguri_dir = r"C:\Users\최대명\Desktop\투자 관련\매도 방법론 패키지(Python)\SOXL 분석용 데이터"
    
    # CSV 파일 목록 가져오기 (대소문자 모두 포함)
    csv_files = glob.glob(os.path.join(jikbakguri_dir, "*.csv"))
    
    if not csv_files:
        print(f"❌ 직박구리 폴더에서 CSV 파일을 찾을 수 없습니다: {jikbakguri_dir}")
        sys.exit(1)
    
    print(f"\n📁 발견된 CSV 파일: {len(csv_files)}개")
    for csv_file in csv_files:
        print(f"  - {os.path.basename(csv_file)}")
    
    # 분석 설정
    max_combinations = 100000000
    n_cores = None
    chunk_size = 50  # 메모리 효율과 CPU 사용률의 균형
    
    # 각 파일별로 분석 실행 및 결과 출력
    all_results = []
    for file_idx, csv_path in enumerate(csv_files, 1):
        print(f"\n\n{'#'*80}")
        print(f"파일 {file_idx}/{len(csv_files)}: {os.path.basename(csv_path)}")
        print(f"{'#'*80}")
        
        result = analyze_single_file(csv_path, max_combinations, n_cores, chunk_size)
        if result:
            all_results.append(result)
            optimizer = result['optimizer']
    
    # 5. 결과 분석
    summary = optimizer.get_performance_summary()
    print(f"\n📊 최적화 성과 요약:")
    print(f"  총 조합 수: {summary['total_combinations']}")
    print(f"  최고 점수: {summary['best_score']:.4f}")
    print(f"  최고 수익률: {summary['best_return']:.2%}")
    print(f"  최고 매도 효율성: {summary['best_efficiency']:.2%}")
    
    # 최적 파라미터 (상위 5개만 sell_records 포함하여 상세 분석)
    best_params = optimizer.get_best_parameters(top_n=5, include_sell_records=True)
    
    print(f"\n🏆 상위 5개 파라미터 조합:")
    for i, param_result in enumerate(best_params):
        print(f"\n{i+1}위 - 종합 점수: {param_result['combined_score']:.4f}")
        print(f"  수익률: {param_result['performance']['total_return']:.2%}")
        print(f"  매도 효율성: {param_result['performance']['sell_efficiency']:.2%}")
        print(f"  매도 횟수: {param_result['performance']['num_trades']}")
        print(f"  평균 신호 강도: {param_result['performance']['avg_signal_strength']:.3f}")
        
        # 주요 파라미터 출력
        params = param_result['params']
        print(f"  주요 파라미터:")
        print(f"    - RSI 과매수: {params.get('rsi_overbought', 'N/A')}, 기간: {params.get('rsi_period', 'N/A')}, 가중치: {params.get('rsi_weight', 'N/A')}")
        print(f"    - OBV 기간: {params.get('obv_period', 'N/A')}, 가중치: {params.get('obv_weight', 'N/A')}")
        print(f"    - ATR 배수: {params.get('atr_multiplier', 'N/A')}, 가중치: {params.get('atr_weight', 'N/A')}")
        print(f"    - 볼린저밴드 임계값: {params.get('bb_position_threshold', 'N/A')}, 기간: {params.get('bb_period', 'N/A')}일, 시그마: {params.get('bb_std_dev', 'N/A')}, 가중치: {params.get('bb_weight', 'N/A')}")
        print(f"    - EMA 기간: 20일선 고정, {params.get('ema_period', 'N/A')}일선")
        print(f"    - 최대 매도 비율: {params.get('max_sell_ratio', 'N/A')} (1% 고정)")
        print(f"    - 매도 가중치 베이스: {params.get('sell_weight_base', 'N/A')}")
    
    # 최고 성과 전략 상세 분석
    if best_params:
        print(f"\n📈 최고 성과 전략 상세 분석:")
        best_result = best_params[0]
        
        print(f"  종합 점수: {best_result['combined_score']:.4f}")
        print(f"  총 수익률: {best_result['performance']['total_return']:.2%}")
        print(f"  매도 효율성: {best_result['performance']['sell_efficiency']:.2%}")
        print(f"  매도 횟수: {best_result['performance']['num_trades']}회")
        print(f"  평균 신호 강도: {best_result['performance']['avg_signal_strength']:.3f}")
        print(f"  최종 보유 비중: {best_result['performance']['final_position']:.2f}%")
        
        # 매도 기록 상세 분석
        if best_result['sell_records']:
            df_sells = pd.DataFrame(best_result['sell_records'])
            
            # 매도 횟수별로 정렬
            df_sells = df_sells.sort_values('sell_count')
                
            print(f"\n  📋 매도 기록 요약:")
            print(f"    - 기술적 지표 기반 매도: {len(df_sells[df_sells['signal_type'] == 'technical'])}회")
            print(f"    - EMA 하향돌파 매도: {len(df_sells[df_sells['signal_type'] == 'ema'])}회")
            print(f"    - 평균 매도 가격: ${df_sells['price'].mean():.2f}")
            print(f"    - 최고 매도 가격: ${df_sells['price'].max():.2f}")
            print(f"    - 최저 매도 가격: ${df_sells['price'].min():.2f}")
    
            print(f"\n  📊 매도 과정 상세 (매도 횟수별):")
            print(f"  {'='*120}")
            # 제목: 각 컬럼의 폭을 정확히 맞춤 (시간가중치, 보유일수 추가)
            print(f"  {'회차':<3} {'날짜':<6} {'매도가격':<6} {'전체기준매도':<7} {'보유기준매도':<7} {'정규화값':<6} {'매도가중치':<6} {'수익률가중치':<7} {'시간가중치':<6} {'보유일수':<5} {'신호유형':<6} {'매도후보유':<7}")
            print(f"  {'-'*120}")
            
            # 파라미터 가져오기
            params = best_result.get('params', {})
            max_sell_ratio = params.get('max_sell_ratio', 0.02)  # 2% 고정
            sell_weight_base = params.get('sell_weight_base', 1.05)
            sell_weight_coefficient = params.get('sell_weight_coefficient', 0.1)
            max_possible_score = (
                params.get('rsi_weight', 1.0) +
                params.get('obv_weight', 1.0) +
                params.get('atr_weight', 1.0) +
                params.get('bb_weight', 1.0) +
                1.0  # EMA는 고정 1.0
            )
            
            current_position = 100.0  # 초기 보유 비중
            
            for idx, row in df_sells.iterrows():
                    sell_count = int(row['sell_count'])
                    date = row['date']
                    price = row['price']
                    ratio = row['ratio']  # 전 매도 후 보유 기준 매도 비율
                    amount = row['amount']  # 전체 100% 기준 매도 비중 (%)
                    signal_strength = row['signal_strength']
                    signal_type = row['signal_type']
                    signal_indicators = row.get('signal_indicators', 'N/A')  # 신호 발생 지표
                    
                    # 매도 가중치 계산 (coefficient * sell_weight_base^(k-1))
                    # 주의: sell_records의 sell_count는 매도 후 값이므로, 실제 계산에 사용된 값은 sell_count - 1
                    actual_sell_count = sell_count - 1 if sell_count > 0 else 0
                    if actual_sell_count == 0:
                        sell_weight = sell_weight_coefficient
                    else:
                        sell_weight = sell_weight_coefficient * (sell_weight_base ** (actual_sell_count - 1))
                    
                    # 정규화값 계산
                    if signal_type == 'technical' and signal_strength > 0:
                        normalized_score = min(signal_strength / max_possible_score, 1.0)
                    elif signal_type == 'ema':
                        normalized_score = 0.1  # EMA 신호는 정규화값 0.1
                    else:
                        normalized_score = 1.0  # 기타 신호
                    
                    # 수익률 가중치 가져오기 (매도 기록에 저장된 값)
                    price_weight = row.get('price_weight', 1.0)
                    
                    # 시간 가중치 및 보유일수 가져오기 (매도 기록에 저장된 값)
                    time_weight = row.get('time_weight', 1.0)
                    days_held = row.get('days_held', 0)
                    
                    # 전체 100% 기준 매도 비율 계산
                    total_based_ratio = amount / 100.0  # amount는 이미 % 단위이므로 100으로 나눔
                    
                    # 보유기준 매도 비율 = 이번 매도량 / 매도 전 보유 비중
                    # 매도 전 보유 비중 = 현재 보유 비중 (아직 차감 전)
                    position_before_sell = current_position
                    hold_based_ratio = amount / position_before_sell if position_before_sell > 0 else 0
                    
                    # 매도 후 보유 비중
                    current_position -= amount
                    
                    # 날짜 포맷팅
                    if hasattr(date, 'strftime'):
                        date_str = date.strftime('%Y-%m-%d')
                    else:
                        date_str = str(date)
                    
                    # 각 컬럼의 폭을 제목과 정확히 맞춤 (일자 정렬)
                    price_str = f"${price:.2f}"
                    total_ratio_str = f"{total_based_ratio:.2%}"  # 전체 100% 기준 매도 비율
                    hold_ratio_str = f"{hold_based_ratio:.2%}"  # 보유 기준 매도 비율 (실제 값)
                    norm_str = f"{normalized_score:.4f}"
                    weight_str = f"{sell_weight:.4f}"
                    price_weight_str = f"{price_weight:.4f}"
                    time_weight_str = f"{time_weight:.4f}"
                    days_held_str = f"{days_held}"
                    position_str = f"{current_position:.2f}%"
                    
                    print(f"  {sell_count:<6} {date_str:<12} {price_str:<12} {total_ratio_str:<14} {hold_ratio_str:<14} {norm_str:<12} {weight_str:<12} {price_weight_str:<14} {time_weight_str:<12} {days_held_str:<10} {signal_strength:<10.4f} {signal_indicators:<14} {position_str:<14}")
            
            print(f"  {'='*180}")
            print(f"  최종 보유 비중: {current_position:.2f}%")
            print(f"  총 매도 비중: {100.0 - current_position:.2f}%")
            
            # 그래프는 모든 파일 분석이 끝난 후에 표시 (데이터만 저장)
            result['df_sells'] = df_sells
    
    # 전체 요약 및 각 파일별 최고 결과 정리
    print("\n\n" + "=" * 80)
    print("🎉 모든 파일 분석이 완료되었습니다!")
    print("=" * 80)
    
    print(f"\n📊 전체 분석 요약:")
    print(f"  분석 완료된 파일: {len(all_results)}개")
    
    # 각 파일별 최고 결과 정리
    print(f"\n🏆 각 파일별 최고 결과 정리 (가중수익률 = 수익률 × 매도비율 기준):")
    print(f"{'='*80}")
    print(f"{'파일명':<15} {'최저가':<6} {'매수평단':<6} {'수익률':<8} {'매도비율':<8} {'가중수익률':<10} {'매도횟수':<5}")
    print(f"{'-'*80}")
    
    summary_data = []
    for result in all_results:
        optimizer = result['optimizer']
        
        # 가중 수익률 (수익률 × 매도비율) 기준으로 최고 결과 찾기
        if optimizer.results:
            # 모든 결과에 대해 가중 수익률 계산
            weighted_results = []
            for r in optimizer.results:
                performance = r['performance']
                total_return = performance['total_return']
                final_position = performance.get('final_position', 0)
                sell_ratio = (100 - final_position) / 100  # 매도비율 (0~1)
                weighted_return = total_return * sell_ratio  # 가중 수익률
                weighted_results.append({
                    'result': r,
                    'weighted_return': weighted_return,
                    'sell_ratio': sell_ratio
                })
            
            # 가중 수익률 기준으로 정렬
            weighted_results.sort(key=lambda x: x['weighted_return'], reverse=True)
            best_weighted = weighted_results[0]
            best = best_weighted['result']
            sell_ratio = best_weighted['sell_ratio']
            weighted_return = best_weighted['weighted_return']
        else:
            continue
        
        file_name = result['file_name']
        low_price = result['low_price']
        buy_avg_price = result['buy_avg_price']
        best_return = best['performance']['total_return']
        best_efficiency = best['performance']['sell_efficiency']
        num_trades = best['performance']['num_trades']
        combined_score = best['combined_score']
        
        print(f"{file_name:<30} ${low_price:<10.2f} ${buy_avg_price:<10.2f} {best_return:<14.2%} {sell_ratio:<14.2%} {weighted_return:<16.2%} {num_trades:<10}")
        
        summary_data.append({
            'file_name': file_name,
            'low_price': low_price,
            'buy_avg_price': buy_avg_price,
            'best_return': best_return,
            'best_efficiency': best_efficiency,
            'num_trades': num_trades,
            'combined_score': combined_score,
            'sell_ratio': sell_ratio,
            'weighted_return': weighted_return,
            'optimizer': optimizer,
            'best_params': [best],  # 가중 수익률 기준 최고 결과
            'data': result.get('data'),  # 기술적 지표가 포함된 데이터
            'csv_path': result.get('csv_path')  # 원본 CSV 파일 경로
        })
    print(f"{'='*80}")
    
    # 가중수익률 기준 Top 10 표 생성
    print(f"\n🏆 가중수익률 기준 Top 10 전략:")
    print(f"{'='*200}")
    
    # 모든 파일의 모든 결과 수집
    all_top_results = []
    for result in all_results:
        optimizer = result['optimizer']
        # 각 파일에서 상위 10개씩 가져오기 (이미 가중수익률 기준 정렬됨)
        top_params = optimizer.get_best_parameters(top_n=10, include_sell_records=False)
        for param_result in top_params:
            all_top_results.append({
                'file_name': result['file_name'],
                'total_return': param_result['performance']['total_return'],
                'sell_ratio': param_result.get('sell_ratio', 0),
                'weighted_return': param_result.get('weighted_return', 0),
                'sell_efficiency': param_result['performance']['sell_efficiency'],
                'num_trades': param_result['performance']['num_trades'],
                'params': param_result['params']
            })
    
    # 가중수익률 기준으로 정렬
    all_top_results.sort(key=lambda x: x['weighted_return'], reverse=True)
    top_10 = all_top_results[:10]
    
    # 표 헤더
    print(f"{'순위':<3} {'파일명':<13} {'수익률':<6} {'매도비율':<6} {'가중수익률':<8} {'매도효율':<6} {'매도횟수':<5} {'RSI':<10} {'OBV':<8} {'ATR':<8} {'BB':<10} {'EMA':<5} {'가중치베이스':<6}")
    print(f"{'-'*200}")
    
    # Top 10 출력
    for idx, result in enumerate(top_10, 1):
        params = result['params']
        rsi_str = f"{params.get('rsi_overbought', 'N/A')}/{params.get('rsi_period', 'N/A')}/{params.get('rsi_weight', 'N/A')}"
        obv_str = f"{params.get('obv_period', 'N/A')}/{params.get('obv_weight', 'N/A')}"
        atr_str = f"{params.get('atr_multiplier', 'N/A')}/{params.get('atr_weight', 'N/A')}"
        bb_str = f"{params.get('bb_period', 'N/A')}/{params.get('bb_std_dev', 'N/A')}/{params.get('bb_weight', 'N/A')}"
        ema_str = f"{params.get('ema_period', 'N/A')}"
        weight_base = params.get('sell_weight_base', 'N/A')
        
        print(f"{idx:<6} {result['file_name']:<26} {result['total_return']:<10.2%} {result['sell_ratio']:<10.2%} {result['weighted_return']:<12.2%} {result['sell_efficiency']:<10.2%} {result['num_trades']:<10} {rsi_str:<20} {obv_str:<16} {atr_str:<16} {bb_str:<20} {ema_str:<10} {weight_base:<12}")
    
    print(f"{'='*90}")
    
    # 각 파일별 Top 10 전략 표시
    print(f"\n🏆 각 파일별 가중수익률 기준 Top 10 전략:")
    for file_idx, result in enumerate(all_results, 1):
        optimizer = result['optimizer']
        file_name = result['file_name']
        
        # 각 파일에서 상위 10개씩 가져오기 (이미 가중수익률 기준 정렬됨)
        top_params = optimizer.get_best_parameters(top_n=10, include_sell_records=False)
        
        if top_params:
            print(f"\n{'#'*110}")
            print(f"📁 파일 {file_idx}: {file_name}")
            print(f"{'#'*110}")
            
            # 표 헤더
            print(f"{'순위':<3} {'수익률':<6} {'매도비율':<6} {'가중수익률':<8} {'매도효율':<6} {'매도횟수':<5} {'RSI':<10} {'OBV':<8} {'ATR':<8} {'BB':<10} {'EMA':<5} {'가중치베이스':<6}")
            print(f"{'-'*110}")
            
            # Top 10 출력
            for rank, param_result in enumerate(top_params, 1):
                params = param_result['params']
                rsi_str = f"{params.get('rsi_overbought', 'N/A')}/{params.get('rsi_period', 'N/A')}/{params.get('rsi_weight', 'N/A')}"
                obv_str = f"{params.get('obv_period', 'N/A')}/{params.get('obv_weight', 'N/A')}"
                atr_str = f"{params.get('atr_multiplier', 'N/A')}/{params.get('atr_weight', 'N/A')}"
                bb_str = f"{params.get('bb_period', 'N/A')}/{params.get('bb_std_dev', 'N/A')}/{params.get('bb_weight', 'N/A')}"
                ema_str = f"{params.get('ema_period', 'N/A')}"
                weight_base = params.get('sell_weight_base', 'N/A')
                sell_ratio = param_result.get('sell_ratio', 0)
                weighted_return = param_result.get('weighted_return', 0)
                
                print(f"{rank:<6} {param_result['performance']['total_return']:<10.2%} {sell_ratio:<10.2%} {weighted_return:<12.2%} {param_result['performance']['sell_efficiency']:<10.2%} {param_result['performance']['num_trades']:<10} {rsi_str:<20} {obv_str:<16} {atr_str:<16} {bb_str:<20} {ema_str:<10} {weight_base:<12}")
            
            print(f"{'='*110}")
    
    # 각 파일별 Top 10 전략을 CSV로 저장
    print(f"\n💾 각 파일별 Top 10 전략 CSV 저장 중...")
    os.makedirs('results', exist_ok=True)
    
    # SOXL 및 USD 파일 필터링
    soxl_results = [r for r in all_results if 'SOXL' in r['file_name'].upper() or 'soxl' in r['file_name'].lower()]
    usd_results = [r for r in all_results if 'USD' in r['file_name'].upper() or 'usd' in r['file_name'].lower()]
    # SOXL과 USD가 모두 포함된 경우 중복 제거
    usd_results = [r for r in usd_results if r not in soxl_results]
    non_grouped_results = [r for r in all_results if r not in soxl_results and r not in usd_results]
    
    # SOXL/USD가 아닌 파일들은 기존 방식으로 저장 (가중수익률 기준)
    for result in non_grouped_results:
        optimizer = result['optimizer']
        file_name = result['file_name']
        # 각 파일에서 상위 10개씩 가져오기 (이미 가중수익률 기준 정렬됨)
        top_params = optimizer.get_best_parameters(top_n=10, include_sell_records=False)
        
        if top_params:
            # CSV 데이터 준비
            csv_data = []
            for rank, param_result in enumerate(top_params, 1):
                params = param_result['params']
                csv_data.append({
                    '순위': rank,
                    '수익률': param_result['performance']['total_return'],
                    '매도비율': param_result.get('sell_ratio', 0),
                    '가중수익률': param_result.get('weighted_return', 0),
                    '매도효율': param_result['performance']['sell_efficiency'],
                    '매도횟수': param_result['performance']['num_trades'],
                    'RSI_과매수': params.get('rsi_overbought', 'N/A'),
                    'RSI_기간': params.get('rsi_period', 'N/A'),
                    'RSI_가중치': params.get('rsi_weight', 'N/A'),
                    'OBV_기간': params.get('obv_period', 'N/A'),
                    'OBV_가중치': params.get('obv_weight', 'N/A'),
                    'ATR_배수': params.get('atr_multiplier', 'N/A'),
                    'ATR_가중치': params.get('atr_weight', 'N/A'),
                    'BB_기간': params.get('bb_period', 'N/A'),
                    'BB_시그마': params.get('bb_std_dev', 'N/A'),
                    'BB_가중치': params.get('bb_weight', 'N/A'),
                    'EMA_기간': params.get('ema_period', 'N/A'),
                    '최대매도비율': params.get('max_sell_ratio', 'N/A'),
                    '매도가중치베이스': params.get('sell_weight_base', 'N/A'),
                    '매도가중치계수': params.get('sell_weight_coefficient', 'N/A'),
                    '수익률가중치지수': params.get('price_weight_exponent', 'N/A'),
                    '시간가중치_최대': params.get('time_weight_max', 'N/A'),
                    '시간가중치_중간점': params.get('time_weight_midpoint', 'N/A'),
                    '시간가중치_기울기': params.get('time_weight_slope', 'N/A')
                })
            
            # CSV 저장
            df_top10 = pd.DataFrame(csv_data)
            safe_filename = file_name.replace('.csv', '').replace('.CSV', '')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_path = os.path.join('results', f'{safe_filename}_Top10_전략_{timestamp}.csv')
            
            # 파라미터 그리드 및 데이터 기간 정보 추가
            with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                # 분석 정보
                f.write("=== 분석 정보 ===\n")
                f.write(f"파일명: {file_name}\n")
                f.write(f"파일 타입: {result.get('file_type', 'N/A')}\n")
                
                # 파라미터 그리드 정보
                param_grid = optimizer.create_parameter_grid()
                f.write("\n[파라미터 그리드]\n")
                for param_name, param_values in param_grid.items():
                    if isinstance(param_values, list):
                        f.write(f"{param_name}: {param_values}\n")
                    else:
                        f.write(f"{param_name}: {param_values}\n")
                
                # 데이터 기간 정보
                f.write("\n[데이터 기간]\n")
                date_range = result.get('date_range', 'N/A')
                f.write(f"{date_range}\n")
                
                f.write("\n\n")
                
                # Top 10 전략
                f.write("=== Top 10 전략 ===\n")
                df_top10.to_csv(f, index=False, encoding='utf-8-sig')
            
            print(f"  ✅ {file_name}: {csv_path}")
    
    # SOXL 파일들을 하나의 CSV로 통합 저장
    if soxl_results:
        print(f"\n💾 SOXL 파일 통합 결과 저장 중...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        soxl_csv_path = os.path.join('results', f'SOXL_통합_결과_{timestamp}.csv')
        
        # ExcelWriter를 사용하여 여러 시트로 저장
        from openpyxl import Workbook
        from openpyxl.utils.dataframe import dataframe_to_rows
        
        # 임시로 CSV로 저장하되, 여러 섹션으로 나누어 저장
        all_sections = []
        
        # 1. 모든 SOXL 파일의 Top 10 전략 통합 (가중수익률 기준)
        all_soxl_top10 = []
        for result in soxl_results:
            optimizer = result['optimizer']
            file_name = result['file_name']
            top_params = optimizer.get_best_parameters(top_n=10, include_sell_records=False)
            
            for rank, param_result in enumerate(top_params, 1):
                params = param_result['params']
                performance = param_result['performance']
                
                # 최고점 대비 가중평균매도금액 비율 계산
                peak_price_ratio = None
                if result.get('data') is not None:
                    data = result['data']
                    if 'Close' in data.columns:
                        peak_price = data['Close'].max()
                        avg_price = performance.get('avg_price', 0)
                        if peak_price > 0 and avg_price > 0:
                            peak_price_ratio = (avg_price / peak_price) * 100
                
                all_soxl_top10.append({
                    '파일명': file_name,
                    '순위': rank,
                    '수익률': performance['total_return'],
                    '최고점대비비율(%)': peak_price_ratio if peak_price_ratio is not None else 'N/A',
                    '매도비율': param_result.get('sell_ratio', 0),
                    '가중수익률': param_result.get('weighted_return', 0),
                    '매도효율': performance['sell_efficiency'],
                    '매도횟수': performance['num_trades'],
                    'RSI_과매수': params.get('rsi_overbought', 'N/A'),
                    'RSI_기간': params.get('rsi_period', 'N/A'),
                    'RSI_가중치': params.get('rsi_weight', 'N/A'),
                    'OBV_기간': params.get('obv_period', 'N/A'),
                    'OBV_가중치': params.get('obv_weight', 'N/A'),
                    'ATR_배수': params.get('atr_multiplier', 'N/A'),
                    'ATR_가중치': params.get('atr_weight', 'N/A'),
                    'BB_기간': params.get('bb_period', 'N/A'),
                    'BB_시그마': params.get('bb_std_dev', 'N/A'),
                    'BB_가중치': params.get('bb_weight', 'N/A'),
                    'EMA_기간': params.get('ema_period', 'N/A'),
                    '최대매도비율': params.get('max_sell_ratio', 'N/A'),
                    '매도가중치베이스': params.get('sell_weight_base', 'N/A'),
                    '매도가중치계수': params.get('sell_weight_coefficient', 'N/A'),
                    '수익률가중치지수': params.get('price_weight_exponent', 'N/A'),
                    '시간가중치_최대': params.get('time_weight_max', 'N/A'),
                    '시간가중치_중간점': params.get('time_weight_midpoint', 'N/A'),
                    '시간가중치_기울기': params.get('time_weight_slope', 'N/A')
                })
        
        # 2. 파라미터 빈도수 계산 (전체 통합 및 파일별)
        # 전체 통합 빈도수
        rsi_freq = {}
        obv_freq = {}
        atr_freq = {}
        bb_freq = {}
        ema_freq = {}
        system_freq = {}
        
        # 파일별 빈도수 (딕셔너리: 파일명 -> 빈도수 딕셔너리)
        file_rsi_freq = {}
        file_obv_freq = {}
        file_atr_freq = {}
        file_bb_freq = {}
        file_ema_freq = {}
        file_system_freq = {}
        
        for result in soxl_results:
            optimizer = result['optimizer']
            file_name = result['file_name']
            top_params = optimizer.get_best_parameters(top_n=10, include_sell_records=False)
            
            # 파일별 빈도수 초기화
            if file_name not in file_rsi_freq:
                file_rsi_freq[file_name] = {}
                file_obv_freq[file_name] = {}
                file_atr_freq[file_name] = {}
                file_bb_freq[file_name] = {}
                file_ema_freq[file_name] = {}
                file_system_freq[file_name] = {}
            
            for param_result in top_params:
                params = param_result['params']
                
                # RSI 조합: 과매수/기간/가중치
                rsi_key = f"{params.get('rsi_overbought', 'N/A')}/{params.get('rsi_period', 'N/A')}/{params.get('rsi_weight', 'N/A')}"
                rsi_freq[rsi_key] = rsi_freq.get(rsi_key, 0) + 1
                file_rsi_freq[file_name][rsi_key] = file_rsi_freq[file_name].get(rsi_key, 0) + 1
                
                # OBV 조합: 기간/가중치
                obv_key = f"{params.get('obv_period', 'N/A')}/{params.get('obv_weight', 'N/A')}"
                obv_freq[obv_key] = obv_freq.get(obv_key, 0) + 1
                file_obv_freq[file_name][obv_key] = file_obv_freq[file_name].get(obv_key, 0) + 1
                
                # ATR 조합: 배수/가중치
                atr_key = f"{params.get('atr_multiplier', 'N/A')}/{params.get('atr_weight', 'N/A')}"
                atr_freq[atr_key] = atr_freq.get(atr_key, 0) + 1
                file_atr_freq[file_name][atr_key] = file_atr_freq[file_name].get(atr_key, 0) + 1
                
                # BB 조합: 기간/시그마/가중치
                bb_key = f"{params.get('bb_period', 'N/A')}/{params.get('bb_std_dev', 'N/A')}/{params.get('bb_weight', 'N/A')}"
                bb_freq[bb_key] = bb_freq.get(bb_key, 0) + 1
                file_bb_freq[file_name][bb_key] = file_bb_freq[file_name].get(bb_key, 0) + 1
                
                # EMA: 기간만
                ema_key = f"{params.get('ema_period', 'N/A')}"
                ema_freq[ema_key] = ema_freq.get(ema_key, 0) + 1
                file_ema_freq[file_name][ema_key] = file_ema_freq[file_name].get(ema_key, 0) + 1
                
                # 시스템 파라미터 조합: 매도가중치계수/매도가중치베이스/수익률가중치지수/시간가중치최대/시간가중치중간점/시간가중치기울기
                system_key = f"{params.get('sell_weight_coefficient', 'N/A')}/{params.get('sell_weight_base', 'N/A')}/{params.get('price_weight_exponent', 'N/A')}/{params.get('time_weight_max', 'N/A')}/{params.get('time_weight_midpoint', 'N/A')}/{params.get('time_weight_slope', 'N/A')}"
                system_freq[system_key] = system_freq.get(system_key, 0) + 1
                file_system_freq[file_name][system_key] = file_system_freq[file_name].get(system_key, 0) + 1
        
        # 빈도수 데이터프레임 생성 (전체 통합)
        rsi_freq_df = pd.DataFrame(list(rsi_freq.items()), columns=['RSI_파라미터(과매수/기간/가중치)', '빈도수']).sort_values('빈도수', ascending=False)
        obv_freq_df = pd.DataFrame(list(obv_freq.items()), columns=['OBV_파라미터(기간/가중치)', '빈도수']).sort_values('빈도수', ascending=False)
        atr_freq_df = pd.DataFrame(list(atr_freq.items()), columns=['ATR_파라미터(배수/가중치)', '빈도수']).sort_values('빈도수', ascending=False)
        bb_freq_df = pd.DataFrame(list(bb_freq.items()), columns=['BB_파라미터(기간/시그마/가중치)', '빈도수']).sort_values('빈도수', ascending=False)
        ema_freq_df = pd.DataFrame(list(ema_freq.items()), columns=['EMA_파라미터(기간)', '빈도수']).sort_values('빈도수', ascending=False)
        system_freq_df = pd.DataFrame(list(system_freq.items()), columns=['시스템_파라미터(매도가중치계수/베이스/수익률지수/시간최대/시간중간점/시간기울기)', '빈도수']).sort_values('빈도수', ascending=False)
        
        # CSV로 저장 (여러 섹션을 구분자로 나누어 저장)
        with open(soxl_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            # 0. 분석 정보 (파라미터 그리드 및 데이터 기간)
            f.write("=== 분석 정보 ===\n")
            f.write("파일 타입: SOXL\n")
            
            # 파라미터 그리드 정보 (첫 번째 결과에서 가져오기)
            if soxl_results:
                first_optimizer = soxl_results[0]['optimizer']
                param_grid = first_optimizer.create_parameter_grid()
                f.write("\n[파라미터 그리드]\n")
                for param_name, param_values in param_grid.items():
                    if isinstance(param_values, list):
                        f.write(f"{param_name}: {param_values}\n")
                    else:
                        f.write(f"{param_name}: {param_values}\n")
            
            # 데이터 기간 정보
            f.write("\n[데이터 기간]\n")
            for result in soxl_results:
                file_name = result['file_name']
                date_range = result.get('date_range', 'N/A')
                f.write(f"{file_name}: {date_range}\n")
            
            f.write("\n\n")
            
            # 1. Top 10 전략
            if all_soxl_top10:
                df_top10 = pd.DataFrame(all_soxl_top10)
                f.write("=== SOXL 파일 통합 Top 10 전략 (가중수익률 기준) ===\n")
                df_top10.to_csv(f, index=False, encoding='utf-8-sig')
                f.write("\n\n")
            
            # 2. 파라미터 빈도수 (종류별 조합)
            f.write("=== RSI 파라미터 빈도수 (과매수/기간/가중치) ===\n")
            rsi_freq_df.to_csv(f, index=False, encoding='utf-8-sig')
            f.write("\n\n")
            
            f.write("=== OBV 파라미터 빈도수 (기간/가중치) ===\n")
            obv_freq_df.to_csv(f, index=False, encoding='utf-8-sig')
            f.write("\n\n")
            
            f.write("=== ATR 파라미터 빈도수 (배수/가중치) ===\n")
            atr_freq_df.to_csv(f, index=False, encoding='utf-8-sig')
            f.write("\n\n")
            
            f.write("=== BB 파라미터 빈도수 (기간/시그마/가중치) ===\n")
            bb_freq_df.to_csv(f, index=False, encoding='utf-8-sig')
            f.write("\n\n")
            
            f.write("=== EMA 파라미터 빈도수 (기간) ===\n")
            ema_freq_df.to_csv(f, index=False, encoding='utf-8-sig')
            f.write("\n\n")
            
            f.write("=== 시스템 파라미터 빈도수 (매도가중치계수/베이스/수익률지수/시간최대/시간기울기) ===\n")
            system_freq_df.to_csv(f, index=False, encoding='utf-8-sig')
            f.write("\n\n")
            
            # 2-1. 파일별 파라미터 빈도수
            f.write("=== 파일별 파라미터 빈도수 ===\n")
            for file_name in sorted(file_rsi_freq.keys()):
                f.write(f"\n--- {file_name} ---\n")
                
                # RSI 빈도수
                if file_rsi_freq[file_name]:
                    file_rsi_df = pd.DataFrame(list(file_rsi_freq[file_name].items()), 
                                               columns=['RSI_파라미터(과매수/기간/가중치)', '빈도수']).sort_values('빈도수', ascending=False)
                    f.write("[RSI 파라미터 빈도수]\n")
                    file_rsi_df.to_csv(f, index=False, encoding='utf-8-sig')
                    f.write("\n")
                
                # OBV 빈도수
                if file_obv_freq[file_name]:
                    file_obv_df = pd.DataFrame(list(file_obv_freq[file_name].items()), 
                                              columns=['OBV_파라미터(기간/가중치)', '빈도수']).sort_values('빈도수', ascending=False)
                    f.write("[OBV 파라미터 빈도수]\n")
                    file_obv_df.to_csv(f, index=False, encoding='utf-8-sig')
                    f.write("\n")
                
                # ATR 빈도수
                if file_atr_freq[file_name]:
                    file_atr_df = pd.DataFrame(list(file_atr_freq[file_name].items()), 
                                              columns=['ATR_파라미터(배수/가중치)', '빈도수']).sort_values('빈도수', ascending=False)
                    f.write("[ATR 파라미터 빈도수]\n")
                    file_atr_df.to_csv(f, index=False, encoding='utf-8-sig')
                    f.write("\n")
                
                # BB 빈도수
                if file_bb_freq[file_name]:
                    file_bb_df = pd.DataFrame(list(file_bb_freq[file_name].items()), 
                                              columns=['BB_파라미터(기간/시그마/가중치)', '빈도수']).sort_values('빈도수', ascending=False)
                    f.write("[BB 파라미터 빈도수]\n")
                    file_bb_df.to_csv(f, index=False, encoding='utf-8-sig')
                    f.write("\n")
                
                # EMA 빈도수
                if file_ema_freq[file_name]:
                    file_ema_df = pd.DataFrame(list(file_ema_freq[file_name].items()), 
                                              columns=['EMA_파라미터(기간)', '빈도수']).sort_values('빈도수', ascending=False)
                    f.write("[EMA 파라미터 빈도수]\n")
                    file_ema_df.to_csv(f, index=False, encoding='utf-8-sig')
                    f.write("\n")
                
                # 시스템 파라미터 빈도수
                if file_system_freq[file_name]:
                    file_system_df = pd.DataFrame(list(file_system_freq[file_name].items()), 
                                                 columns=['시스템_파라미터(매도가중치계수/베이스/수익률지수/시간최대/시간중간점/시간기울기)', '빈도수']).sort_values('빈도수', ascending=False)
                    f.write("[시스템 파라미터 빈도수]\n")
                    file_system_df.to_csv(f, index=False, encoding='utf-8-sig')
                    f.write("\n")
            
            f.write("\n\n")
            
            # 3. 매도 기록 (터미널 표 형식으로 정리) - Top 10 전략 모두
            f.write("=== Top 10 전략 매도 기록 ===\n")
            for result in soxl_results:
                optimizer = result['optimizer']
                file_name = result['file_name']
                top_with_records = optimizer.get_best_parameters(top_n=10, include_sell_records=True)
                
                for rank, param_result in enumerate(top_with_records, 1):
                    if param_result.get('sell_records'):
                        f.write(f"\n--- {file_name} - 순위 {rank} ---\n")
                        df_sells = pd.DataFrame(param_result['sell_records'])
                        df_sells = df_sells.sort_values('sell_count')
                        
                        # 파라미터 가져오기
                        params = param_result.get('params', {})
                        max_sell_ratio = params.get('max_sell_ratio', 0.01)
                        sell_weight_base = params.get('sell_weight_base', 1.05)
                        sell_weight_coefficient = params.get('sell_weight_coefficient', 0.1)
                        max_possible_score = (
                            params.get('rsi_weight', 1.0) +
                            params.get('obv_weight', 1.0) +
                            params.get('atr_weight', 1.0) +
                            params.get('bb_weight', 1.0) +
                            1.0  # EMA는 고정 1.0
                        )
                        
                        # 최고점 가격 계산 (개별 매도 시점 비율 계산용)
                        peak_price = None
                        if result.get('data') is not None:
                            data = result['data']
                            if 'Close' in data.columns:
                                peak_price = data['Close'].max()
                        
                        # 매도 기록 데이터 준비
                        sell_record_data = []
                        current_position = 100.0
                        
                        for idx, row in df_sells.iterrows():
                            sell_count = int(row['sell_count'])
                            date = row['date']
                            price = row['price']
                            ratio = row['ratio']
                            amount = row['amount']
                            signal_strength = row['signal_strength']
                            signal_type = row['signal_type']
                            signal_indicators = row.get('signal_indicators', 'N/A')  # 신호 발생 지표
                            
                            # 매도 가중치 계산 (coefficient * sell_weight_base^(k-1))
                            # 주의: sell_records의 sell_count는 매도 후 값이므로, 실제 계산에 사용된 값은 sell_count - 1
                            actual_sell_count = sell_count - 1 if sell_count > 0 else 0
                            if actual_sell_count == 0:
                                sell_weight = sell_weight_coefficient
                            else:
                                sell_weight = sell_weight_coefficient * (sell_weight_base ** (actual_sell_count - 1))
                            
                            # 정규화값 계산
                            if signal_type == 'technical' and signal_strength > 0:
                                normalized_score = min(signal_strength / max_possible_score, 1.0)
                            elif signal_type == 'ema':
                                normalized_score = 0.1
                            else:
                                normalized_score = 1.0
                            
                            price_weight = row.get('price_weight', 1.0)
                            time_weight = row.get('time_weight', 1.0)
                            days_held = row.get('days_held', 0)
                            total_based_ratio = amount / 100.0
                            
                            # 보유기준 매도 비율 = 이번 매도량 / 매도 전 보유 비중
                            # 매도 전 보유 비중 = 현재 보유 비중 (아직 차감 전)
                            position_before_sell = current_position
                            hold_based_ratio = amount / position_before_sell if position_before_sell > 0 else 0
                            
                            # 날짜 포맷팅
                            if hasattr(date, 'strftime'):
                                date_str = date.strftime('%Y-%m-%d')
                            else:
                                date_str = str(date)
                            
                            # 각 매도 시점의 최고점 대비 매도가격 비율 (backtest_engine에서 저장한 값 사용)
                            sell_peak_ratio = row.get('peak_price_ratio', None)
                            if sell_peak_ratio is None and peak_price is not None and peak_price > 0:
                                sell_peak_ratio = (price / peak_price) * 100
                            
                            current_position -= amount
                            
                            sell_record_data.append({
                                '회차': sell_count,
                                '날짜': date_str,
                                '매도가격': f"${price:.2f}",
                                '전체기준매도': f"{total_based_ratio:.2%}",
                                '보유기준매도': f"{hold_based_ratio:.2%}",
                                '최대매도비율': f"{max_sell_ratio:.2%}",
                                '정규화값': f"{normalized_score:.4f}",
                                '매도가중치': f"{sell_weight:.4f}",
                                '수익률가중치': f"{price_weight:.4f}",
                                '시간가중치': f"{time_weight:.4f}",
                                '보유일수': days_held,
                                '신호강도': f"{signal_strength:.4f}",
                                '신호지표': signal_indicators,
                                '신호유형': signal_type,
                                '매도후보유': f"{current_position:.2f}%",
                                '최고점대비매도가격비율(%)': f"{sell_peak_ratio:.2f}%" if sell_peak_ratio is not None else 'N/A'
                            })
                        
                        if sell_record_data:
                            df_records = pd.DataFrame(sell_record_data)
                            df_records.to_csv(f, index=False, encoding='utf-8-sig')
                            f.write("\n")
        
        print(f"  ✅ SOXL 통합 결과: {soxl_csv_path}")
    
    # USD 파일들을 하나의 CSV로 통합 저장
    if usd_results:
        print(f"\n💾 USD 파일 통합 결과 저장 중...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        usd_csv_path = os.path.join('results', f'USD_통합_결과_{timestamp}.csv')
        
        # 1. 모든 USD 파일의 Top 10 전략 통합 (가중수익률 기준)
        all_usd_top10 = []
        for result in usd_results:
            optimizer = result['optimizer']
            file_name = result['file_name']
            top_params = optimizer.get_best_parameters(top_n=10, include_sell_records=False)
            
            for rank, param_result in enumerate(top_params, 1):
                params = param_result['params']
                performance = param_result['performance']
                
                # 최고점 대비 가중평균매도금액 비율 계산
                peak_price_ratio = None
                if result.get('data') is not None:
                    data = result['data']
                    if 'Close' in data.columns:
                        peak_price = data['Close'].max()
                        avg_price = performance.get('avg_price', 0)
                        if peak_price > 0 and avg_price > 0:
                            peak_price_ratio = (avg_price / peak_price) * 100
                
                all_usd_top10.append({
                    '파일명': file_name,
                    '순위': rank,
                    '수익률': performance['total_return'],
                    '매도비율': param_result.get('sell_ratio', 0),
                    '가중수익률': param_result.get('weighted_return', 0),
                    '매도효율': performance['sell_efficiency'],
                    '매도횟수': performance['num_trades'],
                    '최고점대비매도가격비율(%)': peak_price_ratio if peak_price_ratio is not None else 'N/A',
                    'RSI_과매수': params.get('rsi_overbought', 'N/A'),
                    'RSI_기간': params.get('rsi_period', 'N/A'),
                    'RSI_가중치': params.get('rsi_weight', 'N/A'),
                    'OBV_기간': params.get('obv_period', 'N/A'),
                    'OBV_가중치': params.get('obv_weight', 'N/A'),
                    'ATR_배수': params.get('atr_multiplier', 'N/A'),
                    'ATR_가중치': params.get('atr_weight', 'N/A'),
                    'BB_기간': params.get('bb_period', 'N/A'),
                    'BB_시그마': params.get('bb_std_dev', 'N/A'),
                    'BB_가중치': params.get('bb_weight', 'N/A'),
                    'EMA_기간': params.get('ema_period', 'N/A'),
                    '최대매도비율': params.get('max_sell_ratio', 'N/A'),
                    '매도가중치베이스': params.get('sell_weight_base', 'N/A'),
                    '매도가중치계수': params.get('sell_weight_coefficient', 'N/A'),
                    '수익률가중치지수': params.get('price_weight_exponent', 'N/A'),
                    '시간가중치_최대': params.get('time_weight_max', 'N/A'),
                    '시간가중치_중간점': params.get('time_weight_midpoint', 'N/A'),
                    '시간가중치_기울기': params.get('time_weight_slope', 'N/A')
                })
        
        # 2. 파라미터 빈도수 계산 (전체 통합 및 파일별)
        # 전체 통합 빈도수
        rsi_freq = {}
        obv_freq = {}
        atr_freq = {}
        bb_freq = {}
        ema_freq = {}
        system_freq = {}
        
        # 파일별 빈도수 (딕셔너리: 파일명 -> 빈도수 딕셔너리)
        file_rsi_freq = {}
        file_obv_freq = {}
        file_atr_freq = {}
        file_bb_freq = {}
        file_ema_freq = {}
        file_system_freq = {}
        
        for result in usd_results:
            optimizer = result['optimizer']
            file_name = result['file_name']
            top_params = optimizer.get_best_parameters(top_n=10, include_sell_records=False)
            
            # 파일별 빈도수 초기화
            if file_name not in file_rsi_freq:
                file_rsi_freq[file_name] = {}
                file_obv_freq[file_name] = {}
                file_atr_freq[file_name] = {}
                file_bb_freq[file_name] = {}
                file_ema_freq[file_name] = {}
                file_system_freq[file_name] = {}
            
            for param_result in top_params:
                params = param_result['params']
                
                # RSI 조합: 과매수/기간/가중치
                rsi_key = f"{params.get('rsi_overbought', 'N/A')}/{params.get('rsi_period', 'N/A')}/{params.get('rsi_weight', 'N/A')}"
                rsi_freq[rsi_key] = rsi_freq.get(rsi_key, 0) + 1
                file_rsi_freq[file_name][rsi_key] = file_rsi_freq[file_name].get(rsi_key, 0) + 1
                
                # OBV 조합: 기간/가중치
                obv_key = f"{params.get('obv_period', 'N/A')}/{params.get('obv_weight', 'N/A')}"
                obv_freq[obv_key] = obv_freq.get(obv_key, 0) + 1
                file_obv_freq[file_name][obv_key] = file_obv_freq[file_name].get(obv_key, 0) + 1
                
                # ATR 조합: 배수/가중치
                atr_key = f"{params.get('atr_multiplier', 'N/A')}/{params.get('atr_weight', 'N/A')}"
                atr_freq[atr_key] = atr_freq.get(atr_key, 0) + 1
                file_atr_freq[file_name][atr_key] = file_atr_freq[file_name].get(atr_key, 0) + 1
                
                # BB 조합: 기간/시그마/가중치
                bb_key = f"{params.get('bb_period', 'N/A')}/{params.get('bb_std_dev', 'N/A')}/{params.get('bb_weight', 'N/A')}"
                bb_freq[bb_key] = bb_freq.get(bb_key, 0) + 1
                file_bb_freq[file_name][bb_key] = file_bb_freq[file_name].get(bb_key, 0) + 1
                
                # EMA: 기간만
                ema_key = f"{params.get('ema_period', 'N/A')}"
                ema_freq[ema_key] = ema_freq.get(ema_key, 0) + 1
                file_ema_freq[file_name][ema_key] = file_ema_freq[file_name].get(ema_key, 0) + 1
                
                # 시스템 파라미터 조합: 매도가중치계수/매도가중치베이스/수익률가중치지수/시간가중치최대/시간가중치중간점/시간가중치기울기
                system_key = f"{params.get('sell_weight_coefficient', 'N/A')}/{params.get('sell_weight_base', 'N/A')}/{params.get('price_weight_exponent', 'N/A')}/{params.get('time_weight_max', 'N/A')}/{params.get('time_weight_midpoint', 'N/A')}/{params.get('time_weight_slope', 'N/A')}"
                system_freq[system_key] = system_freq.get(system_key, 0) + 1
                file_system_freq[file_name][system_key] = file_system_freq[file_name].get(system_key, 0) + 1
        
        # 빈도수 데이터프레임 생성 (전체 통합)
        rsi_freq_df = pd.DataFrame(list(rsi_freq.items()), columns=['RSI_파라미터(과매수/기간/가중치)', '빈도수']).sort_values('빈도수', ascending=False)
        obv_freq_df = pd.DataFrame(list(obv_freq.items()), columns=['OBV_파라미터(기간/가중치)', '빈도수']).sort_values('빈도수', ascending=False)
        atr_freq_df = pd.DataFrame(list(atr_freq.items()), columns=['ATR_파라미터(배수/가중치)', '빈도수']).sort_values('빈도수', ascending=False)
        bb_freq_df = pd.DataFrame(list(bb_freq.items()), columns=['BB_파라미터(기간/시그마/가중치)', '빈도수']).sort_values('빈도수', ascending=False)
        ema_freq_df = pd.DataFrame(list(ema_freq.items()), columns=['EMA_파라미터(기간)', '빈도수']).sort_values('빈도수', ascending=False)
        system_freq_df = pd.DataFrame(list(system_freq.items()), columns=['시스템_파라미터(매도가중치계수/베이스/수익률지수/시간최대/시간중간점/시간기울기)', '빈도수']).sort_values('빈도수', ascending=False)
        
        # CSV로 저장 (여러 섹션을 구분자로 나누어 저장)
        with open(usd_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            # 0. 분석 정보 (파라미터 그리드 및 데이터 기간)
            f.write("=== 분석 정보 ===\n")
            f.write("파일 타입: USD\n")
            
            # 파라미터 그리드 정보 (첫 번째 결과에서 가져오기)
            if usd_results:
                first_optimizer = usd_results[0]['optimizer']
                param_grid = first_optimizer.create_parameter_grid()
                f.write("\n[파라미터 그리드]\n")
                for param_name, param_values in param_grid.items():
                    if isinstance(param_values, list):
                        f.write(f"{param_name}: {param_values}\n")
                    else:
                        f.write(f"{param_name}: {param_values}\n")
            
            # 데이터 기간 정보
            f.write("\n[데이터 기간]\n")
            for result in usd_results:
                file_name = result['file_name']
                date_range = result.get('date_range', 'N/A')
                f.write(f"{file_name}: {date_range}\n")
            
            f.write("\n\n")
            
            # 1. Top 10 전략
            if all_usd_top10:
                df_top10 = pd.DataFrame(all_usd_top10)
                f.write("=== USD 파일 통합 Top 10 전략 (가중수익률 기준) ===\n")
                df_top10.to_csv(f, index=False, encoding='utf-8-sig')
                f.write("\n\n")
            
            # 2. 파라미터 빈도수 (종류별 조합)
            f.write("=== RSI 파라미터 빈도수 (과매수/기간/가중치) ===\n")
            rsi_freq_df.to_csv(f, index=False, encoding='utf-8-sig')
            f.write("\n\n")
            
            f.write("=== OBV 파라미터 빈도수 (기간/가중치) ===\n")
            obv_freq_df.to_csv(f, index=False, encoding='utf-8-sig')
            f.write("\n\n")
            
            f.write("=== ATR 파라미터 빈도수 (배수/가중치) ===\n")
            atr_freq_df.to_csv(f, index=False, encoding='utf-8-sig')
            f.write("\n\n")
            
            f.write("=== BB 파라미터 빈도수 (기간/시그마/가중치) ===\n")
            bb_freq_df.to_csv(f, index=False, encoding='utf-8-sig')
            f.write("\n\n")
            
            f.write("=== EMA 파라미터 빈도수 (기간) ===\n")
            ema_freq_df.to_csv(f, index=False, encoding='utf-8-sig')
            f.write("\n\n")
            
            f.write("=== 시스템 파라미터 빈도수 (매도가중치계수/베이스/수익률지수/시간최대/시간기울기) ===\n")
            system_freq_df.to_csv(f, index=False, encoding='utf-8-sig')
            f.write("\n\n")
            
            # 2-1. 파일별 파라미터 빈도수
            f.write("=== 파일별 파라미터 빈도수 ===\n")
            for file_name in sorted(file_rsi_freq.keys()):
                f.write(f"\n--- {file_name} ---\n")
                
                # RSI 빈도수
                if file_rsi_freq[file_name]:
                    file_rsi_df = pd.DataFrame(list(file_rsi_freq[file_name].items()), 
                                               columns=['RSI_파라미터(과매수/기간/가중치)', '빈도수']).sort_values('빈도수', ascending=False)
                    f.write("[RSI 파라미터 빈도수]\n")
                    file_rsi_df.to_csv(f, index=False, encoding='utf-8-sig')
                    f.write("\n")
                
                # OBV 빈도수
                if file_obv_freq[file_name]:
                    file_obv_df = pd.DataFrame(list(file_obv_freq[file_name].items()), 
                                              columns=['OBV_파라미터(기간/가중치)', '빈도수']).sort_values('빈도수', ascending=False)
                    f.write("[OBV 파라미터 빈도수]\n")
                    file_obv_df.to_csv(f, index=False, encoding='utf-8-sig')
                    f.write("\n")
                
                # ATR 빈도수
                if file_atr_freq[file_name]:
                    file_atr_df = pd.DataFrame(list(file_atr_freq[file_name].items()), 
                                              columns=['ATR_파라미터(배수/가중치)', '빈도수']).sort_values('빈도수', ascending=False)
                    f.write("[ATR 파라미터 빈도수]\n")
                    file_atr_df.to_csv(f, index=False, encoding='utf-8-sig')
                    f.write("\n")
                
                # BB 빈도수
                if file_bb_freq[file_name]:
                    file_bb_df = pd.DataFrame(list(file_bb_freq[file_name].items()), 
                                              columns=['BB_파라미터(기간/시그마/가중치)', '빈도수']).sort_values('빈도수', ascending=False)
                    f.write("[BB 파라미터 빈도수]\n")
                    file_bb_df.to_csv(f, index=False, encoding='utf-8-sig')
                    f.write("\n")
                
                # EMA 빈도수
                if file_ema_freq[file_name]:
                    file_ema_df = pd.DataFrame(list(file_ema_freq[file_name].items()), 
                                              columns=['EMA_파라미터(기간)', '빈도수']).sort_values('빈도수', ascending=False)
                    f.write("[EMA 파라미터 빈도수]\n")
                    file_ema_df.to_csv(f, index=False, encoding='utf-8-sig')
                    f.write("\n")
                
                # 시스템 파라미터 빈도수
                if file_system_freq[file_name]:
                    file_system_df = pd.DataFrame(list(file_system_freq[file_name].items()), 
                                                 columns=['시스템_파라미터(매도가중치계수/베이스/수익률지수/시간최대/시간중간점/시간기울기)', '빈도수']).sort_values('빈도수', ascending=False)
                    f.write("[시스템 파라미터 빈도수]\n")
                    file_system_df.to_csv(f, index=False, encoding='utf-8-sig')
                    f.write("\n")
            
            f.write("\n\n")
            
            # 3. 매도 기록 (터미널 표 형식으로 정리) - Top 10 전략 모두
            f.write("=== Top 10 전략 매도 기록 ===\n")
            for result in usd_results:
                optimizer = result['optimizer']
                file_name = result['file_name']
                top_with_records = optimizer.get_best_parameters(top_n=10, include_sell_records=True)
                
                for rank, param_result in enumerate(top_with_records, 1):
                    if param_result.get('sell_records'):
                        f.write(f"\n--- {file_name} - 순위 {rank} ---\n")
                        df_sells = pd.DataFrame(param_result['sell_records'])
                        df_sells = df_sells.sort_values('sell_count')
                        
                        # 파라미터 가져오기
                        params = param_result.get('params', {})
                        max_sell_ratio = params.get('max_sell_ratio', 0.01)
                        sell_weight_base = params.get('sell_weight_base', 1.05)
                        sell_weight_coefficient = params.get('sell_weight_coefficient', 0.1)
                        max_possible_score = (
                            params.get('rsi_weight', 1.0) +
                            params.get('obv_weight', 1.0) +
                            params.get('atr_weight', 1.0) +
                            params.get('bb_weight', 1.0) +
                            1.0  # EMA는 고정 1.0
                        )
                        
                        # 최고점 가격 계산 (개별 매도 시점 비율 계산용)
                        peak_price = None
                        if result.get('data') is not None:
                            data = result['data']
                            if 'Close' in data.columns:
                                peak_price = data['Close'].max()
                        
                        # 매도 기록 데이터 준비
                        sell_record_data = []
                        current_position = 100.0
                        
                        for idx, row in df_sells.iterrows():
                            sell_count = int(row['sell_count'])
                            date = row['date']
                            price = row['price']
                            ratio = row['ratio']
                            amount = row['amount']
                            signal_strength = row['signal_strength']
                            signal_type = row['signal_type']
                            signal_indicators = row.get('signal_indicators', 'N/A')  # 신호 발생 지표
                            
                            # 매도 가중치 계산 (coefficient * sell_weight_base^(k-1))
                            # 주의: sell_records의 sell_count는 매도 후 값이므로, 실제 계산에 사용된 값은 sell_count - 1
                            actual_sell_count = sell_count - 1 if sell_count > 0 else 0
                            if actual_sell_count == 0:
                                sell_weight = sell_weight_coefficient
                            else:
                                sell_weight = sell_weight_coefficient * (sell_weight_base ** (actual_sell_count - 1))
                            
                            # 정규화값 계산
                            if signal_type == 'technical' and signal_strength > 0:
                                normalized_score = min(signal_strength / max_possible_score, 1.0)
                            elif signal_type == 'ema':
                                normalized_score = 0.1
                            else:
                                normalized_score = 1.0
                            
                            price_weight = row.get('price_weight', 1.0)
                            time_weight = row.get('time_weight', 1.0)
                            days_held = row.get('days_held', 0)
                            total_based_ratio = amount / 100.0
                            
                            # 보유기준 매도 비율 = 이번 매도량 / 매도 전 보유 비중
                            # 매도 전 보유 비중 = 현재 보유 비중 (아직 차감 전)
                            position_before_sell = current_position
                            hold_based_ratio = amount / position_before_sell if position_before_sell > 0 else 0
                            
                            # 날짜 포맷팅
                            if hasattr(date, 'strftime'):
                                date_str = date.strftime('%Y-%m-%d')
                            else:
                                date_str = str(date)
                            
                            # 각 매도 시점의 최고점 대비 매도가격 비율 (backtest_engine에서 저장한 값 사용)
                            sell_peak_ratio = row.get('peak_price_ratio', None)
                            if sell_peak_ratio is None and peak_price is not None and peak_price > 0:
                                sell_peak_ratio = (price / peak_price) * 100
                            
                            current_position -= amount
                            
                            sell_record_data.append({
                                '회차': sell_count,
                                '날짜': date_str,
                                '매도가격': f"${price:.2f}",
                                '전체기준매도': f"{total_based_ratio:.2%}",
                                '보유기준매도': f"{hold_based_ratio:.2%}",
                                '최대매도비율': f"{max_sell_ratio:.2%}",
                                '정규화값': f"{normalized_score:.4f}",
                                '매도가중치': f"{sell_weight:.4f}",
                                '수익률가중치': f"{price_weight:.4f}",
                                '시간가중치': f"{time_weight:.4f}",
                                '보유일수': days_held,
                                '신호강도': f"{signal_strength:.4f}",
                                '신호지표': signal_indicators,
                                '신호유형': signal_type,
                                '매도후보유': f"{current_position:.2f}%",
                                '최고점대비매도가격비율(%)': f"{sell_peak_ratio:.2f}%" if sell_peak_ratio is not None else 'N/A'
                            })
                        
                        if sell_record_data:
                            df_records = pd.DataFrame(sell_record_data)
                            df_records.to_csv(f, index=False, encoding='utf-8-sig')
                            f.write("\n")
        
        print(f"  ✅ USD 통합 결과: {usd_csv_path}")
    
    # 파라미터 조합별 두 파일 수익률 비교 CSV 생성 (SOXL/USD 구분 없이 모든 파일 대상)
    if len(all_results) >= 2:
        print(f"\n💾 파라미터 조합별 두 파일 수익률 비교 CSV 저장 중...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 파일 타입 자동 감지 (첫 번째 파일 기준)
        first_file_type = all_results[0].get('file_type', 'Unknown')
        comparison_csv_path = os.path.join('results', f'{first_file_type}_파라미터조합별_수익률비교_{timestamp}.csv')
        
        # 모든 파라미터 조합 수집 (각 파일의 모든 결과)
        param_combinations = {}  # key: 파라미터 조합 문자열, value: {file_name: {performance, peak_price_ratio}}
        
        for result in all_results:
            optimizer = result['optimizer']
            file_name = result['file_name']
            data = result.get('data')
            
            # 모든 파라미터 조합 결과 가져오기 (모든 결과)
            # optimizer.results에 모든 결과가 저장되어 있음
            all_params = optimizer.results if hasattr(optimizer, 'results') and optimizer.results else []
            
            for param_result in all_params:
                params = param_result['params']
                performance = param_result['performance']
                
                # 파라미터 조합을 문자열로 변환 (키 생성)
                param_key = (
                    f"RSI({params.get('rsi_overbought', 'N/A')}/{params.get('rsi_period', 'N/A')}/{params.get('rsi_weight', 'N/A')})_"
                    f"OBV({params.get('obv_period', 'N/A')}/{params.get('obv_weight', 'N/A')})_"
                    f"ATR({params.get('atr_multiplier', 'N/A')}/{params.get('atr_weight', 'N/A')})_"
                    f"BB({params.get('bb_period', 'N/A')}/{params.get('bb_std_dev', 'N/A')}/{params.get('bb_weight', 'N/A')})_"
                    f"EMA({params.get('ema_period', 'N/A')})_"
                    f"시스템({params.get('max_sell_ratio', 'N/A')}/{params.get('sell_weight_base', 'N/A')}/"
                    f"{params.get('price_weight_exponent', 'N/A')}/{params.get('sell_weight_coefficient', 'N/A')})_"
                    f"시간가중치({params.get('time_weight_max', 'N/A')}/{params.get('time_weight_midpoint', 'N/A')}/{params.get('time_weight_slope', 'N/A')})"
                )
                
                if param_key not in param_combinations:
                    param_combinations[param_key] = {}
                
                # 최고점 대비 가중평균매도금액 비율 계산
                peak_price_ratio = None
                if data is not None and 'Close' in data.columns:
                    peak_price = data['Close'].max()
                    avg_price = performance.get('avg_price', 0)
                    if peak_price > 0 and avg_price > 0:
                        peak_price_ratio = (avg_price / peak_price) * 100
                
                # 매도비율 계산 (100 - 최종보유비중) / 100
                final_position = performance.get('final_position', 0)
                sell_ratio = (100 - final_position) / 100  # 매도비율 (0~1)
                
                # 같은 파라미터 조합이 이미 있으면 더 좋은 결과만 유지 (수익률 기준)
                if file_name not in param_combinations[param_key]:
                    param_combinations[param_key][file_name] = {
                        'total_return': performance['total_return'],
                        'peak_price_ratio': peak_price_ratio,
                        'sell_ratio': sell_ratio,  # 매도비율 추가
                        'params': params
                    }
                else:
                    # 더 높은 수익률을 가진 결과로 업데이트
                    if performance['total_return'] > param_combinations[param_key][file_name]['total_return']:
                        param_combinations[param_key][file_name] = {
                            'total_return': performance['total_return'],
                            'peak_price_ratio': peak_price_ratio,
                            'sell_ratio': sell_ratio,  # 매도비율 추가
                            'params': params
                        }
        
        # 두 파일 모두에 결과가 있는 파라미터 조합만 필터링
        file_names = [r['file_name'] for r in all_results]
        valid_combinations = {}
        for param_key, file_results in param_combinations.items():
            # 두 파일 모두에 결과가 있는지 확인
            if len(file_names) == 2 and all(fn in file_results for fn in file_names):
                valid_combinations[param_key] = file_results
        
        # 수익률 합계 계산 및 정렬
        comparison_data = []
        for param_key, file_results in valid_combinations.items():
            # 파라미터 정보 추출 (첫 번째 파일에서)
            first_file = file_names[0]
            params = file_results[first_file]['params']
            
            # 각 파일의 수익률, 최고점 대비 비율, 매도비율
            file1_return = file_results[file_names[0]]['total_return']
            file1_peak_ratio = file_results[file_names[0]]['peak_price_ratio']
            file1_sell_ratio = file_results[file_names[0]].get('sell_ratio', 1.0)  # 매도비율 (기본값 1.0)
            
            file2_return = file_results[file_names[1]]['total_return']
            file2_peak_ratio = file_results[file_names[1]]['peak_price_ratio']
            file2_sell_ratio = file_results[file_names[1]].get('sell_ratio', 1.0)  # 매도비율 (기본값 1.0)
            
            # 수익률 합계
            total_return_sum = file1_return + file2_return
            
            # 평균 수익률
            avg_return = (file1_return + file2_return) / 2
            
            # 수익률 편차 (절댓값 차이)
            return_deviation = abs(file1_return - file2_return)
            
            # 최고점 대비 비율 합계 및 편차 계산 (매도비율 가중 적용)
            # 매도비율이 50%면 최고점대비비율에 0.5를 곱해서 계산
            if file1_peak_ratio is not None and file2_peak_ratio is not None:
                # 매도비율을 곱한 가중 최고점대비비율
                file1_weighted_peak_ratio = file1_peak_ratio * file1_sell_ratio
                file2_weighted_peak_ratio = file2_peak_ratio * file2_sell_ratio
                peak_ratio_sum = file1_weighted_peak_ratio + file2_weighted_peak_ratio
                peak_ratio_deviation = abs(file1_weighted_peak_ratio - file2_weighted_peak_ratio)
            else:
                peak_ratio_sum = None
                peak_ratio_deviation = None
                file1_weighted_peak_ratio = None
                file2_weighted_peak_ratio = None
            
            comparison_data.append({
                '파라미터조합': param_key,
                'RSI_과매수': params.get('rsi_overbought', 'N/A'),
                'RSI_기간': params.get('rsi_period', 'N/A'),
                'RSI_가중치': params.get('rsi_weight', 'N/A'),
                'OBV_기간': params.get('obv_period', 'N/A'),
                'OBV_가중치': params.get('obv_weight', 'N/A'),
                'ATR_배수': params.get('atr_multiplier', 'N/A'),
                'ATR_가중치': params.get('atr_weight', 'N/A'),
                'BB_기간': params.get('bb_period', 'N/A'),
                'BB_시그마': params.get('bb_std_dev', 'N/A'),
                'BB_가중치': params.get('bb_weight', 'N/A'),
                'EMA_기간': params.get('ema_period', 'N/A'),
                '최대매도비율': params.get('max_sell_ratio', 'N/A'),
                '매도가중치베이스': params.get('sell_weight_base', 'N/A'),
                '수익률가중치지수': params.get('price_weight_exponent', 'N/A'),
                '매도가중치계수': params.get('sell_weight_coefficient', 'N/A'),
                '시간가중치_최대': params.get('time_weight_max', 'N/A'),
                '시간가중치_중간점': params.get('time_weight_midpoint', 'N/A'),
                '시간가중치_기울기': params.get('time_weight_slope', 'N/A'),
                f'{file_names[0]}_수익률': file1_return,
                f'{file_names[0]}_최고점대비비율(%)': file1_peak_ratio if file1_peak_ratio is not None else 'N/A',
                f'{file_names[0]}_매도비율': file1_sell_ratio,
                f'{file_names[0]}_가중최고점대비비율(%)': file1_weighted_peak_ratio if file1_weighted_peak_ratio is not None else 'N/A',
                f'{file_names[1]}_수익률': file2_return,
                f'{file_names[1]}_최고점대비비율(%)': file2_peak_ratio if file2_peak_ratio is not None else 'N/A',
                f'{file_names[1]}_매도비율': file2_sell_ratio,
                f'{file_names[1]}_가중최고점대비비율(%)': file2_weighted_peak_ratio if file2_weighted_peak_ratio is not None else 'N/A',
                '수익률합계': total_return_sum,
                '평균수익률': avg_return,
                '수익률편차': return_deviation,
                '가중최고점대비비율합계': peak_ratio_sum if peak_ratio_sum is not None else 'N/A',
                '가중최고점대비비율편차': peak_ratio_deviation if peak_ratio_deviation is not None else 'N/A'
            })
        
        # 가중 최고점 대비 비율 합 - 편차 값 계산
        for combo in comparison_data:
            if combo['가중최고점대비비율합계'] != 'N/A' and combo['가중최고점대비비율편차'] != 'N/A':
                combo['가중최고점대비비율합계_편차차이'] = combo['가중최고점대비비율합계'] - combo['가중최고점대비비율편차']
            else:
                combo['가중최고점대비비율합계_편차차이'] = None
        
        # 가중 최고점 대비 비율 합계가 유효한 데이터만 필터링
        valid_comparison_data = [c for c in comparison_data if c['가중최고점대비비율합계'] != 'N/A']
        
        # 가중 최고점 대비 비율 합계 기준으로 내림차순 정렬
        comparison_data_by_sum = sorted(valid_comparison_data, key=lambda x: x['가중최고점대비비율합계'], reverse=True)
        
        # 상위 100개만 선택 (최고점 대비 비율 합계 기준)
        top50_by_sum = comparison_data_by_sum[:100]
        
        # 가중 최고점 대비 비율 합 - 편차 기준으로 내림차순 정렬 (유효한 값만)
        valid_robust_data = [c for c in valid_comparison_data if c['가중최고점대비비율합계_편차차이'] is not None]
        comparison_data_by_robust = sorted(valid_robust_data, key=lambda x: x['가중최고점대비비율합계_편차차이'], reverse=True)
        
        # 상위 100개만 선택 (최고점 대비 비율 합 - 편차 기준)
        top50_by_robust = comparison_data_by_robust[:100]
        
        # CSV로 저장
        with open(comparison_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            # 가중 최고점 대비 비율 합계 기준 Top 100
            f.write("=== 파라미터 조합별 두 파일 비교 (가중 최고점대비비율 합 상위 100개) ===\n")
            f.write(f"비교 파일: {file_names[0]} vs {file_names[1]}\n")
            f.write("* 가중 최고점대비비율 = 최고점대비비율 × 매도비율 (매도비율이 50%면 0.5를 곱함)\n")
            f.write("\n")
            
            if top50_by_sum:
                df_comparison_sum = pd.DataFrame(top50_by_sum)
                df_comparison_sum.to_csv(f, index=False, encoding='utf-8-sig')
            else:
                f.write("비교 가능한 파라미터 조합이 없습니다.\n")
            
            f.write("\n\n")
            
            # 가중 최고점 대비 비율 합 - 편차 기준 Top 100
            f.write("=== 파라미터 조합별 두 파일 비교 (가중 최고점대비비율 합 - 편차 상위 100개) ===\n")
            f.write(f"비교 파일: {file_names[0]} vs {file_names[1]}\n")
            f.write("* 가중 최고점대비비율 합 - 편차: 높은 가중매도비율과 낮은 편차를 동시에 고려한 강건성 지표\n")
            f.write("* 가중 최고점대비비율 = 최고점대비비율 × 매도비율 (매도비율이 50%면 0.5를 곱함)\n")
            f.write("\n")
            
            if top50_by_robust:
                df_comparison_robust = pd.DataFrame(top50_by_robust)
                df_comparison_robust.to_csv(f, index=False, encoding='utf-8-sig')
            else:
                f.write("비교 가능한 파라미터 조합이 없습니다.\n")
            
            f.write("\n\n")
            
            # === 파라미터 빈도수 분석 ===
            def analyze_param_frequency(data_list, section_name):
                """파라미터 빈도수 분석 함수"""
                rsi_freq = {}
                obv_freq = {}
                atr_freq = {}
                bb_freq = {}
                ema_freq = {}
                system_freq = {}
                time_weight_freq = {}
                
                for item in data_list:
                    # RSI 조합
                    rsi_key = f"{item['RSI_과매수']}/{item['RSI_기간']}/{item['RSI_가중치']}"
                    rsi_freq[rsi_key] = rsi_freq.get(rsi_key, 0) + 1
                    
                    # OBV 조합
                    obv_key = f"{item['OBV_기간']}/{item['OBV_가중치']}"
                    obv_freq[obv_key] = obv_freq.get(obv_key, 0) + 1
                    
                    # ATR 조합
                    atr_key = f"{item['ATR_배수']}/{item['ATR_가중치']}"
                    atr_freq[atr_key] = atr_freq.get(atr_key, 0) + 1
                    
                    # BB 조합
                    bb_key = f"{item['BB_기간']}/{item['BB_시그마']}/{item['BB_가중치']}"
                    bb_freq[bb_key] = bb_freq.get(bb_key, 0) + 1
                    
                    # EMA
                    ema_key = f"{item['EMA_기간']}"
                    ema_freq[ema_key] = ema_freq.get(ema_key, 0) + 1
                    
                    # 시스템 파라미터 (매도가중치계수/매도가중치베이스/수익률가중치지수)
                    system_key = f"{item['매도가중치계수']}/{item['매도가중치베이스']}/{item['수익률가중치지수']}"
                    system_freq[system_key] = system_freq.get(system_key, 0) + 1
                    
                    # 시간가중치 파라미터 (최대/중간점/기울기)
                    time_weight_key = f"{item.get('시간가중치_최대', 'N/A')}/{item.get('시간가중치_중간점', 'N/A')}/{item.get('시간가중치_기울기', 'N/A')}"
                    time_weight_freq[time_weight_key] = time_weight_freq.get(time_weight_key, 0) + 1
                
                return {
                    'RSI': rsi_freq,
                    'OBV': obv_freq,
                    'ATR': atr_freq,
                    'BB': bb_freq,
                    'EMA': ema_freq,
                    '시스템(매도가중치계수/매도가중치베이스/수익률가중치지수)': system_freq,
                    '시간가중치(최대/중간점/기울기)': time_weight_freq
                }
            
            # 가중 최고점 대비 비율 합 Top 100 파라미터 빈도수
            f.write("=== 가중 최고점대비비율 합 Top 100 파라미터 빈도수 ===\n\n")
            sum_freq = analyze_param_frequency(top50_by_sum, "가중 최고점대비비율 합")
            
            for param_type, freq_dict in sum_freq.items():
                sorted_freq = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
                df_freq = pd.DataFrame(sorted_freq, columns=[f'{param_type}_파라미터', '빈도수'])
                f.write(f"[{param_type} 파라미터 빈도수]\n")
                df_freq.to_csv(f, index=False, encoding='utf-8-sig')
                f.write("\n")
            
            f.write("\n\n")
            
            # 가중 최고점 대비 비율 합 - 편차 Top 100 파라미터 빈도수
            f.write("=== 가중 최고점대비비율 합 - 편차 Top 100 파라미터 빈도수 ===\n\n")
            robust_freq = analyze_param_frequency(top50_by_robust, "가중 최고점대비비율 합 - 편차")
            
            for param_type, freq_dict in robust_freq.items():
                sorted_freq = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
                df_freq = pd.DataFrame(sorted_freq, columns=[f'{param_type}_파라미터', '빈도수'])
                f.write(f"[{param_type} 파라미터 빈도수]\n")
                df_freq.to_csv(f, index=False, encoding='utf-8-sig')
                f.write("\n")
            
            f.write("\n\n")
            
            # 두 Top 100 종합 파라미터 빈도수
            f.write("=== 종합 파라미터 빈도수 (Top 100 + Top 100 = 200개 기준) ===\n")
            f.write("* 두 기준(최고점 대비 비율 합, 최고점 대비 비율 합 - 편차) 모두에서 자주 등장하는 파라미터가 강건한 파라미터입니다.\n\n")
            
            combined_list = top50_by_sum + top50_by_robust
            combined_freq = analyze_param_frequency(combined_list, "종합")
            
            for param_type, freq_dict in combined_freq.items():
                sorted_freq = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
                df_freq = pd.DataFrame(sorted_freq, columns=[f'{param_type}_파라미터', '빈도수'])
                f.write(f"[{param_type} 파라미터 빈도수]\n")
                df_freq.to_csv(f, index=False, encoding='utf-8-sig')
                f.write("\n")
            
            f.write("\n\n")
            
            # 교집합 파라미터 조합 (두 Top 100 모두에 등장하는 조합)
            f.write("=== 교집합 파라미터 조합 (두 Top 100 모두에 등장) ===\n")
            f.write("* 가중 최고점대비비율 합 Top 100과 가중 최고점대비비율 합 - 편차 Top 100 모두에 등장하는 파라미터 조합입니다.\n")
            f.write("* 이 조합들이 가장 강건한 파라미터입니다.\n\n")
            
            sum_param_keys = set(item['파라미터조합'] for item in top50_by_sum)
            robust_param_keys = set(item['파라미터조합'] for item in top50_by_robust)
            intersection_keys = sum_param_keys & robust_param_keys
            
            if intersection_keys:
                # 교집합에 해당하는 데이터 추출 (최고점 대비 비율 합 기준으로)
                intersection_data = [item for item in top50_by_sum if item['파라미터조합'] in intersection_keys]
                df_intersection = pd.DataFrame(intersection_data)
                f.write(f"교집합 파라미터 조합 수: {len(intersection_keys)}개\n\n")
                df_intersection.to_csv(f, index=False, encoding='utf-8-sig')
            else:
                f.write("교집합에 해당하는 파라미터 조합이 없습니다.\n")
            
            f.write("\n\n")
            
            # === Top 1 파라미터 조합의 매도 기록 저장 (검증용) ===
            def get_sell_records_for_params(params_dict, all_results_data):
                """파라미터 조합에 대한 매도 기록을 가져오는 함수"""
                from backtest_engine import BacktestEngine
                from sell_signal_generator import SellSignalGenerator
                
                all_sell_records = {}
                
                for result in all_results_data:
                    file_name = result['file_name']
                    data = result.get('data')
                    buy_avg_price = result.get('buy_avg_price')
                    
                    if data is None:
                        continue
                    
                    # 신호 생성기 재생성
                    signal_generator = SellSignalGenerator(data)
                    
                    # 백테스트 실행
                    backtest_engine = BacktestEngine(data, signal_generator)
                    backtest_result = backtest_engine.run_backtest(
                        params_dict,
                        include_sell_records=True,
                        buy_avg_price=buy_avg_price
                    )
                    
                    if backtest_result.get('sell_records'):
                        sell_records = backtest_result['sell_records']
                        performance = backtest_result['performance']
                        all_sell_records[file_name] = {
                            'sell_records': sell_records,
                            'performance': performance
                        }
                
                return all_sell_records
            
            # 가중 최고점대비비율 합 Top 1 매도 기록
            if top50_by_sum:
                f.write("=== 가중 최고점대비비율 합 Top 1 매도 기록 (검증용) ===\n")
                top1_sum = top50_by_sum[0]
                f.write(f"파라미터 조합: {top1_sum['파라미터조합']}\n")
                f.write(f"가중최고점대비비율 합계: {top1_sum['가중최고점대비비율합계']:.2f}%\n\n")
                
                # 파라미터 추출
                top1_sum_params = {
                    'rsi_overbought': top1_sum['RSI_과매수'],
                    'rsi_period': top1_sum['RSI_기간'],
                    'rsi_weight': top1_sum['RSI_가중치'],
                    'obv_period': top1_sum['OBV_기간'],
                    'obv_weight': top1_sum['OBV_가중치'],
                    'atr_multiplier': top1_sum['ATR_배수'],
                    'atr_weight': top1_sum['ATR_가중치'],
                    'bb_position_threshold': 100,
                    'bb_period': top1_sum['BB_기간'],
                    'bb_std_dev': top1_sum['BB_시그마'],
                    'bb_weight': top1_sum['BB_가중치'],
                    'ema_period': top1_sum['EMA_기간'],
                    'max_sell_ratio': top1_sum['최대매도비율'],
                    'sell_weight_base': top1_sum['매도가중치베이스'],
                    'price_weight_exponent': top1_sum['수익률가중치지수'],
                    'sell_weight_coefficient': top1_sum['매도가중치계수'],
                    'time_weight_max': top1_sum['시간가중치_최대'],
                    'time_weight_midpoint': top1_sum['시간가중치_중간점'],
                    'time_weight_slope': top1_sum['시간가중치_기울기']
                }
                
                top1_sum_records = get_sell_records_for_params(top1_sum_params, all_results)
                
                for file_name, record_data in top1_sum_records.items():
                    f.write(f"\n[{file_name} 매도 기록]\n")
                    perf = record_data['performance']
                    f.write(f"수익률: {perf['total_return']:.2%}, 가중평균매도가격: ${perf['avg_price']:.2f}, 총매도비중: {perf.get('total_sold', 0):.2f}%\n")
                    
                    df_sells = pd.DataFrame(record_data['sell_records'])
                    df_sells = df_sells.sort_values('sell_count')
                    # 필요한 컬럼만 선택
                    cols = ['sell_count', 'date', 'price', 'amount', 'signal_type']
                    cols = [c for c in cols if c in df_sells.columns]
                    df_sells[cols].to_csv(f, index=False, encoding='utf-8-sig')
                    f.write("\n")
            
            f.write("\n\n")
            
            # 가중 최고점대비비율 합 - 편차 Top 1 매도 기록
            if top50_by_robust:
                f.write("=== 가중 최고점대비비율 합 - 편차 Top 1 매도 기록 (검증용) ===\n")
                top1_robust = top50_by_robust[0]
                f.write(f"파라미터 조합: {top1_robust['파라미터조합']}\n")
                f.write(f"가중최고점대비비율 합 - 편차: {top1_robust['가중최고점대비비율합계_편차차이']:.2f}%\n\n")
                
                # 파라미터 추출
                top1_robust_params = {
                    'rsi_overbought': top1_robust['RSI_과매수'],
                    'rsi_period': top1_robust['RSI_기간'],
                    'rsi_weight': top1_robust['RSI_가중치'],
                    'obv_period': top1_robust['OBV_기간'],
                    'obv_weight': top1_robust['OBV_가중치'],
                    'atr_multiplier': top1_robust['ATR_배수'],
                    'atr_weight': top1_robust['ATR_가중치'],
                    'bb_position_threshold': 100,
                    'bb_period': top1_robust['BB_기간'],
                    'bb_std_dev': top1_robust['BB_시그마'],
                    'bb_weight': top1_robust['BB_가중치'],
                    'ema_period': top1_robust['EMA_기간'],
                    'max_sell_ratio': top1_robust['최대매도비율'],
                    'sell_weight_base': top1_robust['매도가중치베이스'],
                    'price_weight_exponent': top1_robust['수익률가중치지수'],
                    'sell_weight_coefficient': top1_robust['매도가중치계수'],
                    'time_weight_max': top1_robust['시간가중치_최대'],
                    'time_weight_midpoint': top1_robust['시간가중치_중간점'],
                    'time_weight_slope': top1_robust['시간가중치_기울기']
                }
                
                top1_robust_records = get_sell_records_for_params(top1_robust_params, all_results)
                
                for file_name, record_data in top1_robust_records.items():
                    f.write(f"\n[{file_name} 매도 기록]\n")
                    perf = record_data['performance']
                    f.write(f"수익률: {perf['total_return']:.2%}, 가중평균매도가격: ${perf['avg_price']:.2f}, 총매도비중: {perf.get('total_sold', 0):.2f}%\n")
                    
                    df_sells = pd.DataFrame(record_data['sell_records'])
                    df_sells = df_sells.sort_values('sell_count')
                    # 필요한 컬럼만 선택
                    cols = ['sell_count', 'date', 'price', 'amount', 'signal_type']
                    cols = [c for c in cols if c in df_sells.columns]
                    df_sells[cols].to_csv(f, index=False, encoding='utf-8-sig')
                    f.write("\n")
        
        print(f"  ✅ 파라미터 조합별 비교: {comparison_csv_path}")
        print(f"     - 비교 파일: {file_names[0]} vs {file_names[1]}")
        print(f"     - 비교 가능한 조합 수: {len(valid_combinations)}")
        print(f"     - 가중 최고점대비비율 합 기준 Top 100 저장 완료")
        print(f"     - 가중 최고점대비비율 합 - 편차 기준 Top 100 저장 완료")
        print(f"     - 파라미터 빈도수 분석 저장 완료")
        print(f"     - 교집합 파라미터 조합: {len(intersection_keys)}개")
        
        # comparison_data 변수 업데이트 (이후 그래프 생성용)
        comparison_data = top50_by_sum
        
        # 가중 최고점 대비 비율 합이 가장 높은 파라미터 조합으로 각 파일에 대해 그래프 생성
        if comparison_data:
            best_combo = comparison_data[0]  # 가중 최고점 대비 비율 합이 가장 높은 파라미터 조합
            print(f"\n📊 가중 최고점대비비율 합이 가장 높은 파라미터 조합으로 그래프 생성 중...")
            print(f"   - 가중 최고점대비비율 합계: {best_combo['가중최고점대비비율합계']:.2f}%")
            print(f"   - 수익률 합계: {best_combo['수익률합계']:.2%}")
            print(f"   - 가중 최고점대비비율 편차: {best_combo['가중최고점대비비율편차']:.2f}%")
            
            # 최고 파라미터 조합 추출
            best_params = {
                'rsi_overbought': best_combo['RSI_과매수'],
                'rsi_period': best_combo['RSI_기간'],
                'rsi_weight': best_combo['RSI_가중치'],
                'obv_period': best_combo['OBV_기간'],
                'obv_weight': best_combo['OBV_가중치'],
                'atr_multiplier': best_combo['ATR_배수'],
                'atr_weight': best_combo['ATR_가중치'],
                'bb_position_threshold': 100,  # 볼린저밴드 임계값 (고정값)
                'bb_period': best_combo['BB_기간'],
                'bb_std_dev': best_combo['BB_시그마'],
                'bb_weight': best_combo['BB_가중치'],
                'ema_period': best_combo['EMA_기간'],
                'max_sell_ratio': best_combo['최대매도비율'],
                'sell_weight_base': best_combo['매도가중치베이스'],
                'price_weight_exponent': best_combo['수익률가중치지수'],
                'sell_weight_coefficient': best_combo['매도가중치계수'],
                'time_weight_max': best_combo['시간가중치_최대'],
                'time_weight_midpoint': best_combo['시간가중치_중간점'],
                'time_weight_slope': best_combo['시간가중치_기울기']
            }
            
            # 각 파일에 대해 백테스트 실행 및 그래프 생성
            for result in all_results:
                file_name = result['file_name']
                data = result.get('data')
                csv_path = result.get('csv_path')
                date_range = result.get('date_range', 'N/A')
                buy_avg_price = result.get('buy_avg_price')
                
                if data is None:
                    continue
                
                # 신호 생성기 재생성 (데이터가 이미 기술적 지표 포함)
                from sell_signal_generator import SellSignalGenerator
                signal_generator = SellSignalGenerator(data)
                
                # 백테스트 실행
                from backtest_engine import BacktestEngine
                backtest_engine = BacktestEngine(data, signal_generator)
                backtest_result = backtest_engine.run_backtest(
                    best_params,
                    include_sell_records=True,
                    buy_avg_price=buy_avg_price
                )
                
                # 매도 기록이 있으면 그래프 생성
                if backtest_result.get('sell_records'):
                    df_sells = pd.DataFrame(backtest_result['sell_records'])
                    df_sells = df_sells.sort_values('sell_count')
                    
                    # 그래프 생성 및 저장/팝업
                    print(f"\n  📈 {file_name} 그래프 생성 중...")
                    create_sell_count_charts(
                        df_sells,
                        file_name,
                        csv_path=csv_path,
                        date_range=date_range,
                        total_return=None,  # performance에서 가져오도록 함
                        save_dir='graphs',
                        show=True,
                        criteria='최고점대비비율 합계 기준',
                        performance=backtest_result['performance']  # 백테스트 결과 전달
                    )
                else:
                    print(f"\n  ⚠️ {file_name}: 매도 기록이 없어 그래프를 생성하지 않습니다. (최고점대비비율 합계 기준)")
            
            # 가중 최고점 대비 비율 합 - 편차 값이 가장 큰 파라미터 조합으로 그래프 생성
            best_robust_combo = top50_by_robust[0]  # 가중 최고점 대비 비율 합 - 편차가 가장 큰 파라미터 조합
            
            print(f"\n📊 가중 최고점대비비율 합 - 편차가 가장 큰 파라미터 조합으로 그래프 생성 중...")
            print(f"   - 가중 최고점대비비율 합계: {best_robust_combo['가중최고점대비비율합계']:.2f}%")
            print(f"   - 가중 최고점대비비율 편차: {best_robust_combo['가중최고점대비비율편차']:.2f}%")
            print(f"   - 가중 최고점대비비율 합 - 편차: {best_robust_combo['가중최고점대비비율합계_편차차이']:.2f}%")
            
            # 최고 강건 파라미터 조합 추출
            best_robust_params = {
                'rsi_overbought': best_robust_combo['RSI_과매수'],
                'rsi_period': best_robust_combo['RSI_기간'],
                'rsi_weight': best_robust_combo['RSI_가중치'],
                'obv_period': best_robust_combo['OBV_기간'],
                'obv_weight': best_robust_combo['OBV_가중치'],
                'atr_multiplier': best_robust_combo['ATR_배수'],
                'atr_weight': best_robust_combo['ATR_가중치'],
                'bb_position_threshold': 100,  # 볼린저밴드 임계값 (고정값)
                'bb_period': best_robust_combo['BB_기간'],
                'bb_std_dev': best_robust_combo['BB_시그마'],
                'bb_weight': best_robust_combo['BB_가중치'],
                'ema_period': best_robust_combo['EMA_기간'],
                'max_sell_ratio': best_robust_combo['최대매도비율'],
                'sell_weight_base': best_robust_combo['매도가중치베이스'],
                'price_weight_exponent': best_robust_combo['수익률가중치지수'],
                'sell_weight_coefficient': best_robust_combo['매도가중치계수'],
                'time_weight_max': best_robust_combo['시간가중치_최대'],
                'time_weight_midpoint': best_robust_combo['시간가중치_중간점'],
                'time_weight_slope': best_robust_combo['시간가중치_기울기']
            }
            
            # 각 파일에 대해 백테스트 실행 및 그래프 생성
            for result in all_results:
                file_name = result['file_name']
                data = result.get('data')
                csv_path = result.get('csv_path')
                date_range = result.get('date_range', 'N/A')
                buy_avg_price = result.get('buy_avg_price')
                
                if data is None:
                    continue
                
                # 신호 생성기 재생성 (데이터가 이미 기술적 지표 포함)
                from sell_signal_generator import SellSignalGenerator
                signal_generator = SellSignalGenerator(data)
                
                # 백테스트 실행
                from backtest_engine import BacktestEngine
                backtest_engine = BacktestEngine(data, signal_generator)
                backtest_result = backtest_engine.run_backtest(
                    best_robust_params,
                    include_sell_records=True,
                    buy_avg_price=buy_avg_price
                )
                
                # 매도 기록이 있으면 그래프 생성
                if backtest_result.get('sell_records'):
                    df_sells = pd.DataFrame(backtest_result['sell_records'])
                    df_sells = df_sells.sort_values('sell_count')
                    
                    # 그래프 생성 및 저장/팝업
                    print(f"\n  📈 {file_name} (강건 파라미터) 그래프 생성 중...")
                    create_sell_count_charts(
                        df_sells,
                        file_name,
                        csv_path=csv_path,
                        date_range=date_range,
                        total_return=None,  # performance에서 가져오도록 함
                        save_dir='graphs',
                        show=True,
                        criteria='최고점대비비율-편차 기준',
                        performance=backtest_result['performance']  # 백테스트 결과 전달
                    )
                else:
                    print(f"\n  ⚠️ {file_name}: 매도 기록이 없어 그래프를 생성하지 않습니다. (최고점대비비율-편차 기준)")
    elif len(all_results) < 2:
        print(f"\n⚠️  파일이 2개 미만이어서 파라미터 조합별 비교를 수행할 수 없습니다. (현재: {len(all_results)}개)")
    
    # 각 파일별 상세 최고 결과 출력 (가중 수익률 기준)
    print(f"\n📈 각 파일별 최고 결과 상세 (가중수익률 기준):")
    for idx, summary in enumerate(summary_data, 1):
        print(f"\n{'#'*80}")
        print(f"파일 {idx}: {summary['file_name']}")
        print(f"{'#'*80}")
        
        best_result = summary['best_params'][0]
        params = best_result['params']
        
        # 매도비율 및 가중 수익률 계산
        final_position = best_result['performance'].get('final_position', 0)
        sell_ratio = (100 - final_position) / 100
        weighted_return = best_result['performance']['total_return'] * sell_ratio
        
        print(f"  📊 성과 지표:")
        print(f"    - 종합 점수: {best_result['combined_score']:.4f}")
        print(f"    - 총 수익률: {best_result['performance']['total_return']:.2%}")
        print(f"    - 매도비율: {sell_ratio:.2%}")
        print(f"    - 가중 수익률 (수익률×매도비율): {weighted_return:.2%}")
        print(f"    - 매도 효율성: {best_result['performance']['sell_efficiency']:.2%}")
        print(f"    - 매도 횟수: {best_result['performance']['num_trades']}회")
        print(f"    - 평균 신호 강도: {best_result['performance']['avg_signal_strength']:.3f}")
        print(f"    - 최종 보유 비중: {final_position:.2f}%")
        
        print(f"  ⚙️  최적 파라미터:")
        print(f"    - RSI: 과매수={params.get('rsi_overbought', 'N/A')}, 기간={params.get('rsi_period', 'N/A')}, 가중치={params.get('rsi_weight', 'N/A')}")
        print(f"    - OBV: 기간={params.get('obv_period', 'N/A')}, 가중치={params.get('obv_weight', 'N/A')} (고정)")
        print(f"    - ATR: 배수={params.get('atr_multiplier', 'N/A')}, 가중치={params.get('atr_weight', 'N/A')} (고정)")
        print(f"    - EMA: 20일선 고정, {params.get('ema_period', 'N/A')}일선")
        print(f"    - 최소 신호 임계값: {params.get('min_signal_threshold', 'N/A')}")
        print(f"    - 최대 매도 비율: {params.get('max_sell_ratio', 'N/A')}")
        
        # 가중 수익률 기준 최고 결과의 매도 기록 데이터 저장 (나중에 그래프 표시용)
        # best_params[0]에 저장된 파라미터로 다시 백테스트하여 sell_records 가져오기
        from backtest_engine import BacktestEngine
        from sell_signal_generator import SellSignalGenerator
        
        data = summary.get('data')
        if data is not None:
            signal_generator = SellSignalGenerator(data)
            backtest_engine = BacktestEngine(data, signal_generator)
            
            # 최고점 가격 계산
            peak_price = data['Close'].max() if 'Close' in data.columns else None
            
            # 가중 수익률 기준 최고 파라미터로 백테스트 재실행
            backtest_result = backtest_engine.run_backtest(
                best_result['params'],
                include_sell_records=True,
                buy_avg_price=summary.get('buy_avg_price')
            )
            
            if backtest_result.get('sell_records'):
                df_sells = pd.DataFrame(backtest_result['sell_records'])
                df_sells = df_sells.sort_values('sell_count')
                # 그래프 데이터 저장 (나중에 표시하기 위해)
                summary['df_sells'] = df_sells
                summary['backtest_performance'] = backtest_result['performance']  # performance도 저장
    
    # 모든 파일 분석이 끝난 후 그래프 표시
    print(f"\n📊 모든 파일의 그래프 표시:")
    for idx, summary in enumerate(summary_data, 1):
        if 'df_sells' in summary and not summary['df_sells'].empty:
            print(f"\n  📈 파일 {idx}: {summary['file_name']}")
            date_range = summary.get('date_range', 'N/A')
            create_sell_count_charts(
                summary['df_sells'], 
                summary['file_name'], 
                csv_path=summary.get('csv_path'), 
                date_range=date_range, 
                total_return=None,  # performance에서 가져오도록 함
                save_dir='graphs', 
                show=True, 
                criteria='가중수익률 기준',
                performance=summary.get('backtest_performance')  # 백테스트 결과 전달
            )
    
    print("\n💡 다음 단계:")
    print("  1. 각 파일별 상위 파라미터 조합을 검토하고 선택하세요")
    print("  2. 선택한 파라미터로 추가 백테스트를 실행하세요")
    print("  3. 실제 투자 전 충분한 검증을 거치세요")
    print("=" * 80)


