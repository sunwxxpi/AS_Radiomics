import pandas as pd
from sklearn.model_selection import train_test_split
from config import Config

class DataSplitter:
    """데이터셋을 train/test로 분할하는 클래스"""
    
    def __init__(self):
        """Config에서 데이터 분할 설정을 가져옴"""
        self.split_mode = Config.DATA_SPLIT_MODE
        self.random_state = Config.DATA_SPLIT_RANDOM_STATE
        self.test_size = Config.TEST_SIZE_RATIO
    
    def split_data(self, features_df, mode='binary'):
        """
        데이터를 train/test로 분할
        
        Args:
            features_df: 특징이 추출된 DataFrame (severity 열, data_source 열 포함)
            mode: 'binary' 또는 'multi'
            
        Returns:
            train_df, test_df: 분할된 DataFrames
        """
        if self.split_mode == 'fix':
            return self._split_by_directory(features_df, mode)
        else:
            return self._split_randomly(features_df, mode)
        
    def _split_by_directory(self, features_df, mode='binary'):
        """디렉토리 기반 고정 분할 방식"""
        print(f"\n--- 디렉토리 기반 고정 분할 시작 ---")
        print(f"  Train: imagesTr + labelsTr")
        print(f"  Test: imagesVal + labelsVal")
        
        if features_df.empty or 'severity' not in features_df.columns:
            print("  오류: 유효한 데이터프레임이 아닙니다.")
            return pd.DataFrame(), pd.DataFrame()
            
        if 'data_source' not in features_df.columns:
            print("  오류: data_source 열이 필요합니다.")
            return pd.DataFrame(), pd.DataFrame()
        
        # 분할 전 클래스 분포
        print("  분할 전 전체 클래스 분포:")
        pre_split_dist = features_df['severity'].value_counts().to_dict()
        pre_split_dist_pct = features_df['severity'].value_counts(normalize=True).to_dict()
        
        # 클래스 순서대로 출력
        if mode == 'multi':
            class_order = ['normal', 'nonsevere', 'severe']
            available_classes = [cls for cls in class_order if cls in pre_split_dist]
            for cls in available_classes:
                print(f"    {cls}: {pre_split_dist.get(cls, 0)} ({pre_split_dist_pct.get(cls, 0):.1%})")
        else:
            for cls, count in sorted(pre_split_dist.items()):
                print(f"    {cls}: {count} ({pre_split_dist_pct[cls]:.1%})")
        
        # 디렉토리 기반 분할
        train_df = features_df[features_df['data_source'] == 'train'].copy()
        test_df = features_df[features_df['data_source'] == 'val'].copy()
        
        if train_df.empty or test_df.empty:
            print("  경고: 일부 데이터셋이 비어있습니다.")
            if train_df.empty:
                print("    Train 데이터가 없습니다.")
            if test_df.empty:
                print("    Test 데이터가 없습니다.")
        
        # 분할 후 클래스 분포
        print("\n  분할 후 클래스 분포:")
        print(f"  Train ({len(train_df)} 샘플):")
        if not train_df.empty:
            train_dist = train_df['severity'].value_counts().to_dict()
            train_dist_pct = train_df['severity'].value_counts(normalize=True).to_dict()
            
            if mode == 'multi':
                for cls in available_classes:
                    print(f"    {cls}: {train_dist.get(cls, 0)} ({train_dist_pct.get(cls, 0):.1%})")
            else:
                for cls, count in sorted(train_dist.items()):
                    print(f"    {cls}: {count} ({train_dist_pct[cls]:.1%})")
        
        print(f"\n  Test ({len(test_df)} 샘플):")
        if not test_df.empty:
            test_dist = test_df['severity'].value_counts().to_dict()
            test_dist_pct = test_df['severity'].value_counts(normalize=True).to_dict()
            
            if mode == 'multi':
                for cls in available_classes:
                    print(f"    {cls}: {test_dist.get(cls, 0)} ({test_dist_pct.get(cls, 0):.1%})")
            else:
                for cls, count in sorted(test_dist.items()):
                    print(f"    {cls}: {count} ({test_dist_pct[cls]:.1%})")
        
        print(f"\n  데이터 분할 완료: Train {len(train_df)} 샘플 ({len(train_df)/len(features_df):.1%}), "
              f"Test {len(test_df)} 샘플 ({len(test_df)/len(features_df):.1%})")
        
        return train_df, test_df
    
    def _split_randomly(self, features_df, mode='binary'):
        """랜덤 분할 방식"""
        print(f"\n--- 랜덤 데이터 분할 시작 (비율: {100*(1-self.test_size):.0f}:{100*self.test_size:.0f}) ---")
        print(f"  랜덤 시드: {self.random_state}")
        
        if features_df.empty or 'severity' not in features_df.columns:
            print("  오류: 유효한 데이터프레임이 아닙니다.")
            return pd.DataFrame(), pd.DataFrame()
        
        # 분할 전 클래스 분포
        print("  분할 전 클래스 분포:")
        pre_split_dist = features_df['severity'].value_counts().to_dict()
        pre_split_dist_pct = features_df['severity'].value_counts(normalize=True).to_dict()
        
        # 클래스 순서대로 출력
        if mode == 'multi':
            class_order = ['normal', 'nonsevere', 'severe']
            available_classes = [cls for cls in class_order if cls in pre_split_dist]
            for cls in available_classes:
                print(f"    {cls}: {pre_split_dist.get(cls, 0)} ({pre_split_dist_pct.get(cls, 0):.1%})")
        else:
            for cls, count in sorted(pre_split_dist.items()):
                print(f"    {cls}: {count} ({pre_split_dist_pct[cls]:.1%})")
        
        # 데이터 분할 (계층적 샘플링 적용)
        try:
            train_df, test_df = train_test_split(
                features_df,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=features_df['severity']
            )
        except ValueError as e:
            print(f"  오류: 데이터 분할 실패 - {e}")
            print("  일부 클래스에 샘플이 너무 적을 수 있습니다. 일반 분할로 대체합니다.")
            train_df, test_df = train_test_split(
                features_df,
                test_size=self.test_size,
                random_state=self.random_state
            )
        
        # 분할 후 클래스 분포
        print("\n  분할 후 클래스 분포:")
        print(f"  Train ({len(train_df)} 샘플):")
        train_dist = train_df['severity'].value_counts().to_dict()
        train_dist_pct = train_df['severity'].value_counts(normalize=True).to_dict()
        
        if mode == 'multi':
            for cls in available_classes:
                print(f"    {cls}: {train_dist.get(cls, 0)} ({train_dist_pct.get(cls, 0):.1%})")
        else:
            for cls, count in sorted(train_dist.items()):
                print(f"    {cls}: {count} ({train_dist_pct[cls]:.1%})")
        
        print(f"\n  Test ({len(test_df)} 샘플):")
        test_dist = test_df['severity'].value_counts().to_dict()
        test_dist_pct = test_df['severity'].value_counts(normalize=True).to_dict()
        
        if mode == 'multi':
            for cls in available_classes:
                print(f"    {cls}: {test_dist.get(cls, 0)} ({test_dist_pct.get(cls, 0):.1%})")
        else:
            for cls, count in sorted(test_dist.items()):
                print(f"    {cls}: {count} ({test_dist_pct[cls]:.1%})")
        
        print(f"\n  데이터 분할 완료: Train {len(train_df)} 샘플 ({len(train_df)/len(features_df):.1%}), "
              f"Test {len(test_df)} 샘플 ({len(test_df)/len(features_df):.1%})")
        
        return train_df, test_df