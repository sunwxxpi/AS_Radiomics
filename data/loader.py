import pandas as pd
import sys

class DataLoader:
    """데이터 로딩 및 레이블 매핑을 담당하는 클래스"""
    
    def __init__(self, label_file_path):
        self.label_file_path = label_file_path
        self.severity_df = None
        self.severity_map = None
    
    def load_labels(self):
        """레이블 파일 로딩 및 전처리"""
        print(f"  '{self.label_file_path}' 파일 로딩 시도...")
        
        try:
            self.severity_df = pd.read_csv(self.label_file_path)
            
            # 필수 컬럼 확인
            required_columns = ['1차년도연구번호', 'AV_binaryclassification']
            if not all(col in self.severity_df.columns for col in required_columns):
                raise ValueError(f"Label file must contain {required_columns} columns.")
            
            print(f"  '{self.label_file_path}' 로딩 성공.")
            print(f"  로드된 원본 데이터 행 수: {len(self.severity_df)}")
            
            # 결측값 제거
            self.severity_df.dropna(subset=['AV_binaryclassification'], inplace=True)
            print(f"  'AV_binaryclassification' 결측값 제거 후 행 수: {len(self.severity_df)}")
            
            # 매핑 딕셔너리 생성
            self.severity_map = self.severity_df.set_index('1차년도연구번호')['AV_binaryclassification'].to_dict()
            print(f"  매핑 딕셔너리 생성 완료 (총 {len(self.severity_map)} 개 항목)")
            print(f"  고유 AV_binaryclassification 값: {self.severity_df['AV_binaryclassification'].unique()}")
            
            return self.severity_map
            
        except FileNotFoundError:
            print(f"  오류: 레이블 파일 '{self.label_file_path}'을 찾을 수 없습니다.")
            sys.exit(1)
        except Exception as e:
            print(f"  오류: 레이블 파일 처리 중 오류 발생: {e}")
            sys.exit(1)
    
    def get_severity_mapping(self):
        """중증도 매핑 딕셔너리 반환"""
        if self.severity_map is None:
            self.load_labels()
        return self.severity_map