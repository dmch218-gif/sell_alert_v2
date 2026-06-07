# =============================================================================
# 하이퍼파라미터 최적화 시스템 (멀티프로세싱 + TQDM)
# =============================================================================
import pandas as pd
import numpy as np
from itertools import product
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
from backtest_engine import BacktestEngine
from sell_signal_generator import SellSignalGenerator

def evaluate_single_parameter(args):
    """
    단일 파라미터 조합 평가 (멀티프로세싱용)
    
    Args:
        args: (params, data, signal_generator, buy_avg_price) 튜플
    
    Returns:
        평가 결과 딕셔너리 (메모리 효율을 위해 sell_records 제외)
    """
    params, data, signal_generator, buy_avg_price = args
    
    try:
        # 백테스트 엔진 생성
        backtest_engine = BacktestEngine(data, signal_generator)
        
        # 백테스트 실행 (sell_records는 메모리 절약을 위해 제외)
        result = backtest_engine.run_backtest(params, include_sell_records=False, buy_avg_price=buy_avg_price)
        
        # 종합 점수 계산
        performance = result['performance']
        combined_score = (
            performance['total_return'] * 0.4 +
            performance['sell_efficiency'] * 0.3 +
            (performance['avg_signal_strength'] if performance['num_trades'] > 0 else 0) * 0.3
        )
        
        # 메모리 효율을 위해 sell_records는 제외하고 요약 정보만 저장
        return {
            'params': params.copy(),
            'performance': performance,
            'combined_score': combined_score
        }
    except Exception as e:
        print(f"파라미터 평가 중 오류: {e}")
        return None

def evaluate_with_generator(args):
    """프로세스 내에서 signal_generator를 생성하여 평가 (멀티프로세싱용)"""
    params, data, _, buy_avg_price = args
    signal_generator = SellSignalGenerator(data)
    return evaluate_single_parameter((params, data, signal_generator, buy_avg_price))

class HyperparameterOptimizer:
    """하이퍼파라미터 최적화 클래스"""
    
    def __init__(self, data, signal_generator, buy_avg_price=None, file_type='SOXL'):
        """
        Args:
            data: 기술적 지표가 포함된 데이터프레임
            signal_generator: SellSignalGenerator 인스턴스
            buy_avg_price: 매수 평균단가 (None이면 첫 거래일 종가 사용)
            file_type: 파일 타입 ('SOXL' 또는 'USD')
        """
        self.data = data.copy()
        self.signal_generator = signal_generator
        self.buy_avg_price = buy_avg_price
        self.file_type = file_type
        self.results = []
    
    def create_parameter_grid(self):
        """하이퍼파라미터 그리드 생성 (파일 타입에 따라 다른 그리드 사용)"""
        import numpy as np
        
        if self.file_type == 'SOXL':
            # SOXL 확정 파라미터 (2024-12-21 최종 결정)
            param_grid = {
                # RSI 파라미터
                'rsi_overbought': [66],
                'rsi_period': [43],
                'rsi_weight': [9],
                
                # OBV 파라미터
                'obv_period': [57],
                'obv_weight': [3],
                
                # ATR 파라미터
                'atr_multiplier': [2],  # 또는 3 (강건성 동등)
                'atr_weight': [3],
                
                # 볼린저밴드 파라미터 (민감도 낮음)
                'bb_position_threshold': [100],
                'bb_period': [40],  # 35~55 중 중간값
                'bb_std_dev': [3.5],
                'bb_weight': [3],
                
                # EMA 파라미터
                'ema_period': [60],
                
                # 시스템 파라미터
                'max_sell_ratio': [0.01],
                'sell_weight_base': [1.01],
                'price_weight_exponent': [0.4],
                'sell_weight_coefficient': [1.6],
                
                # 시간 가중치 파라미터
                'time_weight_max': [40],
                'time_weight_midpoint': [550],
                'time_weight_slope': [0.045]
            }
        elif self.file_type == 'USD':
            # USD 확정 파라미터 (2024-12-21 최종 결정 - USD_파라미터조합별_수익률비교_20251201_081339 기반)
            param_grid = {
                # RSI 파라미터 (빈도수 1위: 65/48/10, 19회)
                'rsi_overbought': [65],
                'rsi_period': [48],
                'rsi_weight': [10],
                
                # OBV 파라미터 (100% 고정: 60/5.0, 200회)
                'obv_period': [60],
                'obv_weight': [5.0],
                
                # ATR 파라미터 (100% 고정: 3/4.0, 200회)
                'atr_multiplier': [3],
                'atr_weight': [4.0],
                
                # 볼린저밴드 파라미터 (빈도수 1위: 60/2.0/6, 57회)
                'bb_position_threshold': [100],
                'bb_period': [60],
                'bb_std_dev': [2.0],
                'bb_weight': [6],
                
                # EMA 파라미터
                'ema_period': [60],
                
                # 시스템 파라미터 (빈도수 1위: 0.04/1.04/2.0, 143회)
                'max_sell_ratio': [0.01],
                'sell_weight_base': [1.04],
                'price_weight_exponent': [2.0],
                'sell_weight_coefficient': [0.04],
                
                # 시간 가중치 파라미터 (비활성화: time_weight_max=1이면 항상 1)
                'time_weight_max': [1],
                'time_weight_midpoint': [1],
                'time_weight_slope': [0.045]
            }
        else:
            raise ValueError(f"지원하지 않는 파일 타입: {self.file_type}")
        
        return param_grid
    
    def grid_search(self, max_combinations=1000, n_cores=None, chunk_size=50):
        """
        그리드 서치 실행 (멀티프로세싱 + TQDM) - 메모리 효율 개선
        
        Args:
            max_combinations: 최대 조합 수
            n_cores: 사용할 CPU 코어 수 (None이면 자동 선택)
            chunk_size: 청크 크기 (메모리 절약을 위해 작게 설정 권장)
        
        Returns:
            평가 결과 리스트
        """
        param_grid = self.create_parameter_grid()
        
        # 파라미터 조합 생성
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # 모든 조합 생성 (제너레이터로 처리하여 메모리 절약)
        print("🔄 파라미터 조합 생성 중...", flush=True)
        
        # 조합 수 사전 계산 (디버깅용)
        total_combinations = 1
        print("📋 파라미터별 개수:", flush=True)
        for name, values in zip(param_names, param_values):
            count = len(values)
            total_combinations *= count
            print(f"   {name:25s}: {count:6d}개", flush=True)
        print(f"📊 예상 총 조합 수: {total_combinations:,}개", flush=True)
        
        all_combinations_gen = product(*param_values)
        all_combinations = list(all_combinations_gen)
        
        # 실제 생성된 조합 수 확인
        actual_count = len(all_combinations)
        if actual_count != total_combinations:
            print(f"⚠️  경고: 예상 조합 수({total_combinations:,})와 실제 조합 수({actual_count:,})가 다릅니다!", flush=True)
        
        # 최대 조합 수 제한
        if actual_count > max_combinations:
            import random
            random.seed(42)
            print(f"⚠️  조합 수가 {actual_count:,}개로 제한치를 초과하여 {max_combinations:,}개로 샘플링합니다...", flush=True)
            all_combinations = random.sample(all_combinations, max_combinations)
        
        print(f"✅ 총 {len(all_combinations):,}개 조합 생성 완료", flush=True)
        
        # CPU 코어 수 설정
        if n_cores is None:
            n_cores = cpu_count()  # 모든 코어 사용
        else:
            n_cores = min(n_cores, cpu_count())
        
        print(f"⚡ 사용할 CPU 코어 수: {n_cores}개 (전체 {cpu_count()}개 중)", flush=True)
        
        # 청크로 나누기 (CPU 사용률 최대화를 위해 동적 조정)
        # 각 코어가 항상 작업할 수 있도록 충분한 작업 큐 유지
        # 청크 크기를 코어 수의 배수로 조정하여 부하 균형 개선
        optimal_chunk_size = max(chunk_size, n_cores * 2)  # 최소 코어 수의 2배
        # 코어 수의 배수로 조정 (메모리와 성능의 균형)
        if optimal_chunk_size % n_cores != 0:
            # 가장 가까운 코어 수의 배수로 반올림
            optimal_chunk_size = ((optimal_chunk_size + n_cores - 1) // n_cores) * n_cores
        
        print(f"📦 청크 크기: {chunk_size} → 최적화: {optimal_chunk_size} (코어 수 {n_cores}의 배수)", flush=True)
        chunks = [all_combinations[i:i + optimal_chunk_size] for i in range(0, len(all_combinations), optimal_chunk_size)]
        
        start_time = time.time()
        all_results = []
        
        # 전체 진행상황 표시 (Windows 멀티프로세싱 환경에서도 제대로 표시되도록 설정)
        import sys
        # TQDM과 함께 간단한 진행 상황 출력도 추가
        # 변수 초기화 (예외 발생 시에도 사용 가능하도록)
        pbar = None
        use_tqdm = False
        
        try:
            # 초기에는 숨기고, 첫 번째 업데이트가 있을 때 표시
            pbar = tqdm(total=len(all_combinations), desc="", unit="조합", 
                       dynamic_ncols=False, ncols=150, mininterval=0.5, maxinterval=2.0, 
                       file=sys.stdout, disable=False, leave=True, 
                       bar_format='진행률 : {bar} {percentage:3.0f}% {postfix}',
                       initial=0, miniters=1, position=0, ascii=False)
            # 초기 상태 숨기기 (첫 업데이트 전까지는 표시하지 않음)
            pbar.display(False)
            use_tqdm = True
        except Exception:
            # TQDM 실패 시 간단한 출력으로 대체
            pbar = None
            use_tqdm = False
            print(f"\n📊 총 {len(all_combinations):,}개 조합 처리 시작...")
        
        # 데이터를 한 번만 복사하여 메모리 절약
        data_copy = self.data.copy()
        
        # 멀티프로세싱 풀 생성
        print("🚀 멀티프로세싱 시작...", flush=True)
        
        # TQDM이 제대로 표시되도록 초기화
        if use_tqdm and pbar is not None:
            pbar.display(True)  # 시작 시 바로 표시
            pbar.refresh()  # 화면 갱신
        
        with Pool(processes=n_cores) as pool:
            # 각 청크 처리
            for chunk_idx, chunk in enumerate(chunks):
                # 현재 청크에 대한 인수 준비
                args_list = []
                for combination in chunk:
                    params = dict(zip(param_names, combination))
                    
                    # 메모리 효율: 각 프로세스마다 데이터를 복사하지 않고 참조만 전달
                    # (실제로는 프로세스 간 통신 시 pickle로 직렬화되므로 복사가 발생하지만,
                    #  청크 단위로 처리하여 메모리 사용량을 제한)
                    args_list.append((
                        params,
                        data_copy,  # 공유 데이터 참조
                        None,  # signal_generator는 각 프로세스에서 생성
                        self.buy_avg_price  # 매수 평균단가 전달
                    ))
                
                # 현재 청크 처리 (각 프로세스에서 signal_generator 생성)
                # imap 사용으로 메모리 효율 향상 및 CPU 사용률 개선
                # chunksize를 코어 수에 맞게 조정하여 CPU 사용률 최대화
                imap_chunksize = max(1, len(args_list) // (n_cores * 2))  # 각 프로세스가 항상 작업할 수 있도록
                chunk_results = []
                chunk_completed = 0
                
                for result in pool.imap(evaluate_with_generator, args_list, chunksize=imap_chunksize):
                    if result is not None:
                        chunk_results.append(result)
                        all_results.append(result)
                    
                    # 각 결과가 나올 때마다 진행 상황 업데이트
                    chunk_completed += 1
                    completed = len(all_results)
                    
                    # TQDM 업데이트
                    if use_tqdm and pbar is not None:
                        pbar.update(1)
                    
                    # 현재 상태 출력 (매번 업데이트하여 실시간 표시)
                    elapsed_time = time.time() - start_time
                    if completed > 0 and elapsed_time > 0:
                        avg_time_per_combination = elapsed_time / completed
                        remaining_combinations = len(all_combinations) - completed
                        estimated_remaining_time = remaining_combinations * avg_time_per_combination
                        progress_pct = (completed / len(all_combinations)) * 100
                        speed = completed / elapsed_time
                        
                        # 시간 포맷팅 (시간:분:초)
                        elapsed_hour = int(elapsed_time // 3600)
                        elapsed_min = int((elapsed_time % 3600) // 60)
                        elapsed_sec = int(elapsed_time % 60)
                        remaining_hour = int(estimated_remaining_time // 3600)
                        remaining_min = int((estimated_remaining_time % 3600) // 60)
                        remaining_sec = int(estimated_remaining_time % 60)
                        
                        if use_tqdm and pbar is not None:
                            try:
                                # TQDM postfix에 정보 추가 (진행률(%) | 조합 진행률 | 시간 | 속도)
                                pbar.set_postfix_str(
                                    f"진행률: {progress_pct:.1f}% | "
                                    f"조합: {completed:,}/{len(all_combinations):,} | "
                                    f"시간: {elapsed_hour:02d}:{elapsed_min:02d}:{elapsed_sec:02d}/{remaining_hour:02d}:{remaining_min:02d}:{remaining_sec:02d} | "
                                    f"속도: {speed:.1f}조합/초",
                                    refresh=True
                                )
                                pbar.refresh()  # 화면 즉시 갱신
                            except:
                                pass
                        else:
                            # TQDM이 없을 때 print로 출력
                            print(f"\r📊 진행률: {progress_pct:.1f}% | "
                                  f"조합: {completed:,}/{len(all_combinations):,} | "
                                  f"시간: {elapsed_hour:02d}:{elapsed_min:02d}:{elapsed_sec:02d}/{remaining_hour:02d}:{remaining_min:02d}:{remaining_sec:02d} | "
                                  f"속도: {speed:.1f}조합/초", end='', flush=True)
                
                # 청크 처리 완료 후 즉시 메모리 해제
                del chunk_results
                
                # 메모리 정리 (가비지 컬렉션) - CPU 사용률을 위해 주기적으로만 실행
                if chunk_idx % 10 == 0:  # 10개 청크마다 한 번만 실행
                    import gc
                    gc.collect()
        
        if use_tqdm and pbar is not None:
            pbar.close()
        else:
            print()  # 마지막 줄바꿈
            print(f"✅ 완료: {len(all_results):,}개 조합 처리 완료")
        
        # 유효한 결과만 필터링
        self.results = all_results
        
        return self.results
    
    def get_best_parameters(self, top_n=10, include_sell_records=False, sort_by='weighted_return'):
        """
        최적 파라미터 반환
        
        Args:
            top_n: 반환할 상위 결과 수
            include_sell_records: sell_records를 포함할지 여부 (메모리 사용량 증가)
            sort_by: 정렬 기준 
                - 'weighted_return': 가중수익률 (수익률 × 매도비율)
                - 'peak_ratio_weighted_sum': 최고점 대비 매도비율의 합
                - 'total_return': 총 수익률
        
        Returns:
            최적 파라미터 리스트
        """
        if not self.results:
            return []
        
        # 가중수익률 및 추가 지표 계산
        weighted_results = []
        for r in self.results:
            performance = r['performance']
            total_return = performance['total_return']
            final_position = performance.get('final_position', 0)
            sell_ratio = (100 - final_position) / 100  # 매도비율 (0~1)
            weighted_return = total_return * sell_ratio  # 가중 수익률
            peak_ratio_weighted_sum = performance.get('peak_ratio_weighted_sum', 0)  # 최고점 대비 매도비율의 합
            weighted_results.append({
                'result': r,
                'weighted_return': weighted_return,
                'sell_ratio': sell_ratio,
                'peak_ratio_weighted_sum': peak_ratio_weighted_sum,
                'total_return': total_return
            })
        
        # 정렬 기준에 따라 정렬
        if sort_by == 'peak_ratio_weighted_sum':
            weighted_results.sort(key=lambda x: x['peak_ratio_weighted_sum'], reverse=True)
        elif sort_by == 'total_return':
            weighted_results.sort(key=lambda x: x['total_return'], reverse=True)
        else:  # 기본값: weighted_return
            weighted_results.sort(key=lambda x: x['weighted_return'], reverse=True)
        
        top_results = [wr['result'] for wr in weighted_results[:top_n]]
        
        # 추가 정보를 결과에 추가
        for i, wr in enumerate(weighted_results[:top_n]):
            top_results[i]['weighted_return'] = wr['weighted_return']
            top_results[i]['sell_ratio'] = wr['sell_ratio']
            top_results[i]['peak_ratio_weighted_sum'] = wr['peak_ratio_weighted_sum']
        
        # 필요시 상위 결과에 대해 sell_records 재계산
        if include_sell_records and top_results:
            for result in top_results:
                backtest_engine = BacktestEngine(self.data, self.signal_generator)
                detailed_result = backtest_engine.run_backtest(
                    result['params'], 
                    include_sell_records=True,
                    buy_avg_price=self.buy_avg_price
                )
                result['sell_records'] = detailed_result.get('sell_records', [])
        
        return top_results
    
    def get_performance_summary(self):
        """성과 요약 정보 반환"""
        if not self.results:
            return {}
        
        scores = [result['combined_score'] for result in self.results]
        returns = [result['performance']['total_return'] for result in self.results]
        efficiencies = [result['performance']['sell_efficiency'] for result in self.results]
        
        return {
            'total_combinations': len(self.results),
            'best_score': max(scores),
            'worst_score': min(scores),
            'avg_score': np.mean(scores),
            'best_return': max(returns),
            'worst_return': min(returns),
            'avg_return': np.mean(returns),
            'best_efficiency': max(efficiencies),
            'worst_efficiency': min(efficiencies),
            'avg_efficiency': np.mean(efficiencies)
        }

