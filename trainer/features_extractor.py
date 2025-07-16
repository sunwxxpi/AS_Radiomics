import os
import re
import logging
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import binary_dilation, generate_binary_structure
from radiomics import featureextractor
from trainer.dl_embedding_extractor import DLEmbeddingExtractor

class RadiomicsExtractor:
    """Radiomics 특징 추출을 담당하는 클래스"""
    
    def __init__(self, geometry_tolerance=1e-5, enable_dl_embedding=False, dl_model_path=None, dl_model_type='custom', dl_nnunet_config=None,
                 enable_dilation=False, dilation_iterations=1):
        self.extractor = featureextractor.RadiomicsFeatureExtractor()
        self.extractor.settings['geometryTolerance'] = geometry_tolerance
        self.enable_dl_embedding = enable_dl_embedding
        self.dl_extractor = None
        self.enable_dilation = enable_dilation
        self.dilation_iterations = dilation_iterations
        
        # DL embedding 추출기 초기화
        if self.enable_dl_embedding and dl_model_path and os.path.exists(dl_model_path):
            try:
                from config import Config
                self.dl_extractor = DLEmbeddingExtractor(
                    model_path=dl_model_path,
                    model_type=dl_model_type,
                    nnunet_config=dl_nnunet_config,
                    img_size=Config.DL_IMG_SIZE
                )
                print(f"  DL embedding 추출기 초기화 완료 (IMG SIZE: {Config.DL_IMG_SIZE})")
            except Exception as e:
                print(f"  DL embedding 추출기 초기화 실패: {e}")
                self.dl_extractor = None
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Radiomics 로거 설정"""
        logger = logging.getLogger("radiomics")
        logger.setLevel(logging.ERROR)
    
    def _apply_dilation(self, label_path):
        """레이블 마스크에 dilation 적용"""
        try:
            original_img = nib.load(label_path)
            original_data = original_img.get_fdata().astype(np.uint8)
            
            structure = generate_binary_structure(rank=3, connectivity=3)
            dilated_data = binary_dilation(
                input=original_data,
                structure=structure,
                iterations=self.dilation_iterations
            ).astype(original_data.dtype)
            
            dilated_img = nib.Nifti1Image(
                dataobj=dilated_data,
                affine=original_img.affine,
                header=original_img.header
            )
            
            temp_label_path = label_path.replace('.nii.gz', f'_dilated_{self.dilation_iterations}iter_temp.nii.gz')
            nib.save(dilated_img, temp_label_path)
            
            return temp_label_path
            
        except Exception as e:
            print(f"      Dilation 적용 오류: {e}")
            return label_path
    
    def extract_features_for_set(self, image_dir, label_dir, set_name, patient_info_map, mode='binary'):
        """특정 데이터셋에 대한 특징 추출"""
        dilation_info = f"(Dilation: {self.dilation_iterations}회)" if self.enable_dilation else ""
        dl_info = "(+ DL Embedding Fusion)" if self.enable_dl_embedding and self.dl_extractor else ""
        print(f"\n  '{set_name}' 세트 특징 추출 시작 (모드: {mode}) {dl_info} {dilation_info}")
        
        if not os.path.isdir(image_dir):
            print(f"    오류: 이미지 디렉토리를 찾을 수 없음: {image_dir}")
            return pd.DataFrame()
        
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
        print(f"    총 {len(image_files)}개의 .nii.gz 파일 발견")
        
        if not image_files:
            print(f"    경고: 이미지 파일을 찾을 수 없습니다.")
            return pd.DataFrame()
        
        features_list = []
        processed_cases = []
        skipped_cases = []
        
        for image_filename in image_files:
            result = self._process_single_case(
                image_filename, image_dir, label_dir, patient_info_map, set_name)
            
            if result['success']:
                features_list.append(result['features'])
                processed_cases.append(result['case_id'])
            else:
                skipped_cases.append(result['case_id'])
        
        self._print_extraction_summary(set_name, processed_cases, skipped_cases)
        
        if not features_list:
            return pd.DataFrame()
        
        features_df = pd.DataFrame(features_list)
        if 'case_id' in features_df.columns:
            features_df.set_index('case_id', inplace=True)
        
        return features_df
    
    def _process_single_case(self, image_filename, image_dir, label_dir, patient_info_map, set_name):
        """단일 케이스 처리"""
        print(f"\n    [{set_name}] 파일 처리: {image_filename}")
        
        # 파일명 파싱
        match = re.match(r'([A-Za-z0-9\.\-]+)_(\d{4,})_0000\.nii\.gz', image_filename)
        if not match:
            print(f"      건너뛰기: 파일명 형식 불일치")
            return {'success': False, 'case_id': image_filename}
        
        patient_id = match.group(1).strip()
        sequence_part = match.group(2).strip()
        case_id = f"{patient_id}_{sequence_part}"
        
        # 환자 정보 확인
        label = patient_info_map.get(patient_id)
        if label is None:
            print(f"      건너뛰기: ID '{patient_id}'를 환자 정보 맵에서 찾을 수 없음")
            return {'success': False, 'case_id': patient_id}
        
        # 파일 경로 설정
        image_path = os.path.join(image_dir, image_filename)
        label_path = os.path.join(label_dir, f"{case_id}.nii.gz")
        
        if not os.path.exists(label_path):
            print(f"      건너뛰기: 레이블 파일 부재")
            return {'success': False, 'case_id': case_id}
        
        # Dilation 적용 (필요한 경우)
        final_label_path = label_path
        temp_label_path = None
        
        if self.enable_dilation:
            final_label_path = self._apply_dilation(label_path)
            if final_label_path != label_path:
                temp_label_path = final_label_path
        
        # 특징 추출
        try:
            # Radiomics 특징 추출
            result = self.extractor.execute(image_path, final_label_path, label=1)
            features = {key: val for key, val in result.items() if not key.startswith('diagnostics_')}
            
            # DL embedding 특징 추가
            if self.enable_dl_embedding and self.dl_extractor:
                dl_features = self.dl_extractor.extract_features_for_case(image_path, case_id)
                features.update(dl_features)
            
            features['case_id'] = case_id
            features['severity'] = label
            
            # 결과 출력
            radiomics_count = len([k for k in features.keys() 
                                 if not k.startswith('dl_embedding_') and k not in ['case_id', 'severity']])
            dl_count = len([k for k in features.keys() if k.startswith('dl_embedding_')])
            
            if dl_count > 0:
                print(f"      성공: {radiomics_count}개 radiomics + {dl_count}개 DL embedding 특징 추출")
            else:
                print(f"      성공: {radiomics_count}개 radiomics 특징 추출")
            
            # 임시 파일 정리
            if temp_label_path and os.path.exists(temp_label_path):
                try:
                    os.remove(temp_label_path)
                except OSError:
                    pass
            
            return {'success': True, 'case_id': case_id, 'features': features}
            
        except Exception as e:
            print(f"      오류: 특징 추출 실패 - {e}")
            
            # 임시 파일 정리
            if temp_label_path and os.path.exists(temp_label_path):
                try:
                    os.remove(temp_label_path)
                except OSError:
                    pass
            
            return {'success': False, 'case_id': case_id}
    
    def _print_extraction_summary(self, set_name, processed_cases, skipped_cases):
        """추출 결과 요약 출력"""
        print(f"\n    --- '{set_name}' 세트 특징 추출 요약 ---")
        print(f"    성공적으로 처리된 케이스 수: {len(processed_cases)}")
        
        unique_skipped = sorted(list(set(skipped_cases)))
        if unique_skipped:
            print(f"    건너뛴 고유 케이스 수: {len(unique_skipped)}")