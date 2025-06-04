import pandas as pd
import sys

class DataLoader:
    """데이터 로딩 및 레이블 매핑을 담당하는 클래스"""
    
    def __init__(self, label_file_path):
        self.label_file_path = label_file_path
        self.severity_df = None
        self.binary_map = None
        self.multi_map = None
    
    def load_labels(self, mode='binary'):
        """레이블 파일 로딩 및 전처리"""
        if self.severity_df is None:
            self._load_and_preprocess_data()
        
        # 모드에 따라 적절한 맵 반환
        if mode == 'binary':
            if self.binary_map is None:
                self.binary_map = self._create_binary_map()
            return self.binary_map
        elif mode == 'multi':
            if self.multi_map is None:
                self.multi_map = self._create_multi_map()
            return self.multi_map
        else:
            raise ValueError(f"알 수 없는 모드: {mode}. 'binary' 또는 'multi'로 지정하세요.")
    
    def _load_and_preprocess_data(self):
        """레이블 파일 로딩 및 기본 전처리"""
        print(f"  '{self.label_file_path}' 파일 로딩 시도...")
        
        try:
            self.severity_df = pd.read_csv(self.label_file_path)
            
            # 필수 컬럼 확인
            required_columns = ['1차년도연구번호', 'AV_binaryclassification', 'AS ']
            if not all(col in self.severity_df.columns for col in required_columns):
                raise ValueError(f"Label file must contain {required_columns} columns.")
            
            print(f"  '{self.label_file_path}' 로딩 성공.")
            print(f"  로드된 원본 데이터 행 수: {len(self.severity_df)}")
            
            # 'AS ' 컬럼명 변경 및 값 정규화
            self.severity_df.rename(columns={'AS ': 'AS_grade'}, inplace=True)
            
            # AS_grade 값 정규화 (대소문자 무시, 앞뒤 공백 제거, 소문자로 통일)
            self.severity_df['AS_grade'] = self.severity_df['AS_grade'].astype(str).str.strip().str.lower()
            
            # 정규화 후 결측값(nan이 문자열 'nan'으로 변환된 경우) 처리
            self.severity_df['AS_grade'] = self.severity_df['AS_grade'].replace('nan', 'Unknown')
            self.severity_df['AS_grade'] = self.severity_df['AS_grade'].fillna('Unknown')
            
            print(f"  정규화된 고유 AS_grade 값: {sorted(self.severity_df['AS_grade'].unique())}")
            
        except FileNotFoundError:
            print(f"  오류: 레이블 파일 '{self.label_file_path}'을 찾을 수 없습니다.")
            sys.exit(1)
        except Exception as e:
            print(f"  오류: 레이블 파일 처리 중 오류 발생: {e}")
            sys.exit(1)
    
    def _create_binary_map(self):
        """이진 분류를 위한 맵 생성 (AV_binaryclassification 기반)"""
        print(f"  이진 분류 맵 생성 중...")
        
        # 'AV_binaryclassification' 결측값이 있는 행 제거
        binary_df = self.severity_df.dropna(subset=['AV_binaryclassification']).copy()
        print(f"  'AV_binaryclassification' 결측값 제거 후 행 수: {len(binary_df)}")
        
        # 맵 생성
        binary_map = binary_df.set_index('1차년도연구번호')['AV_binaryclassification'].to_dict()
        print(f"  이진 분류 맵 생성 완료 (총 {len(binary_map)} 개 항목)")
        print(f"  고유 AV_binaryclassification 값: {sorted(binary_df['AV_binaryclassification'].unique())}")
        
        return binary_map
    
    def _create_multi_map(self):
        """3-클래스 분류를 위한 맵 생성 (AS_grade 기반)"""
        print(f"  다중 분류 맵 생성 중...")
        
        # AS_grade 값을 3개 클래스로 통합
        multi_df = self.severity_df.copy()
        
        # 3-클래스 변환 함수 정의
        def map_to_three_class(grade):
            grade = str(grade).strip().lower()
            if grade in ['none']:
                return 'normal'
            elif grade in ['mild', 'moderate']:
                return 'nonsevere'
            elif grade in ['severe', 'very severe']:
                return 'severe'
            else:
                return 'unknown'
        
        # 변환 적용
        multi_df['AS_grade_3class'] = multi_df['AS_grade'].apply(map_to_three_class)
        
        # 'unknown' 클래스 제거
        multi_df = multi_df[multi_df['AS_grade_3class'] != 'unknown']
        
        # 맵 생성
        multi_map = multi_df.set_index('1차년도연구번호')['AS_grade_3class'].to_dict()
        
        print(f"  다중 분류 맵 생성 완료 (총 {len(multi_map)} 개 항목)")
        print(f"  3-클래스 AS_grade 값: {sorted(multi_df['AS_grade_3class'].unique())}")
        print(f"  3-클래스 분포: {multi_df['AS_grade_3class'].value_counts()}")
        
        return multi_map
    
    def get_severity_mapping(self, mode='binary'):
        """주어진 모드에 따른 중증도 매핑 반환"""
        return self.load_labels(mode)