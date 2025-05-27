import pandas as pd
import sys

class DataLoader:
    """데이터 로딩 및 레이블 매핑을 담당하는 클래스"""
    
    def __init__(self, label_file_path):
        self.label_file_path = label_file_path
        self.severity_df = None
        self.severity_map = None
        self.patient_info_map = None
    
    def load_labels(self):
        """레이블 파일 로딩 및 전처리"""
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
            
            # 'AV_binaryclassification' 결측값 제거
            self.severity_df.dropna(subset=['AV_binaryclassification'], inplace=True)
            print(f"  'AV_binaryclassification' 결측값 제거 후 행 수: {len(self.severity_df)}")
            
            # 매핑 딕셔너리 생성 (AV_binaryclassification 및 AS_grade 포함)
            self.patient_info_map = self.severity_df.set_index('1차년도연구번호')[
                ['AV_binaryclassification', 'AS_grade']
            ].to_dict('index')
            
            print(f"  환자 정보 맵 생성 완료 (총 {len(self.patient_info_map)} 개 항목)")
            print(f"  고유 AV_binaryclassification 값: {self.severity_df['AV_binaryclassification'].unique()}")
            print(f"  정규화된 고유 AS_grade 값: {sorted(self.severity_df['AS_grade'].unique())}")
            
            return self.patient_info_map
            
        except FileNotFoundError:
            print(f"  오류: 레이블 파일 '{self.label_file_path}'을 찾을 수 없습니다.")
            sys.exit(1)
        except Exception as e:
            print(f"  오류: 레이블 파일 처리 중 오류 발생: {e}")
            sys.exit(1)
    
    def get_severity_mapping(self):
        """중증도 매핑 딕셔너리 반환"""
        if self.patient_info_map is None:
            self.load_labels()
        return self.patient_info_map