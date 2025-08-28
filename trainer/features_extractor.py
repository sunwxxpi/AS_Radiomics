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
    """Radiomics와 DL embedding 특징을 추출하는 통합 클래스
    
    Multiple fold DL 모델을 지원하여 Radiomics는 한 번만 추출하고
    각 fold별 DL embedding만 추출하여 효율성을 높임
    """
    
    def __init__(self, geometry_tolerance=1e-5, enable_dl_embedding=False, dl_model_paths=None, dl_model_type='custom', dl_nnunet_config=None,
                 enable_dilation=False, dilation_iterations=1):
        self.extractor = featureextractor.RadiomicsFeatureExtractor()
        self.extractor.settings['geometryTolerance'] = geometry_tolerance
        self.enable_dl_embedding = enable_dl_embedding
        self.dl_extractors = {}  # fold별 DL embedding 추출기 저장소
        self.enable_dilation = enable_dilation
        self.dilation_iterations = dilation_iterations
        
        # 각 fold별 DL embedding 추출기 초기화
        if self.enable_dl_embedding and dl_model_paths:
            from config import Config
            for fold, model_path in dl_model_paths.items():
                if os.path.exists(model_path):
                    try:
                        self.dl_extractors[fold] = DLEmbeddingExtractor(
                            model_path=model_path,
                            model_type=dl_model_type,
                            nnunet_config=dl_nnunet_config,
                            img_size=Config.DL_IMG_SIZE
                        )
                        print(f"  DL embedding 추출기 초기화 완료 - Fold {fold} (IMG SIZE: {Config.DL_IMG_SIZE})\n")
                    except Exception as e:
                        print(f"  DL embedding 추출기 초기화 실패 - Fold {fold}: {e}\n")
                else:
                    print(f"  경고: Fold {fold} 모델 파일을 찾을 수 없음: {model_path}\n")
        
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
    
    def extract_radiomics_features_for_set(self, image_dir, label_dir, set_name, patient_info_map, mode='binary'):
        """Radiomics 특징만 추출 (DL embedding 제외)
        
        효율성을 위해 Radiomics는 한 번만 추출하고 이후 fold별 DL embedding과 결합
        """
        print(f"\n  '{set_name}' 세트 Radiomics 특징 추출 시작 (모드: {mode})")
        
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
            result = self._extract_radiomics_only(
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
        # case_id를 컬럼으로 유지 (전처리 호환성)
        
        return features_df
    
    def _extract_radiomics_only(self, image_filename, image_dir, label_dir, patient_info_map, set_name):
        """단일 케이스의 Radiomics 특징만 추출
        
        파일명 패턴: {patient_id}_{sequence}_0000.nii.gz -> {patient_id}_{sequence}.nii.gz
        """
        print(f"\n    [{set_name}] Radiomics 특징 추출: {image_filename}")
        
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
        
        # Radiomics 특징 추출
        try:
            result = self.extractor.execute(image_path, final_label_path, label=1)
            features = {key: val for key, val in result.items() if not key.startswith('diagnostics_')}
            
            features['case_id'] = case_id
            features['severity'] = label
            features['image_path'] = image_path  # DL 특징 추출용 경로 저장
            # 데이터 소스 정보 추가 (디렉토리 기반 분할을 위해)
            data_source = 'train' if 'Tr' in set_name else 'val'
            features['data_source'] = data_source
            
            radiomics_count = len([k for k in features.keys() 
                                 if k not in ['case_id', 'severity', 'image_path', 'data_source']])
            
            print(f"      성공: {radiomics_count}개 radiomics 특징 추출")
            
            # 임시 파일 정리
            if temp_label_path and os.path.exists(temp_label_path):
                try:
                    os.remove(temp_label_path)
                except OSError:
                    pass
            
            return {'success': True, 'case_id': case_id, 'features': features}
            
        except Exception as e:
            print(f"      오류: Radiomics 특징 추출 실패 - {e}")
            
            # 임시 파일 정리
            if temp_label_path and os.path.exists(temp_label_path):
                try:
                    os.remove(temp_label_path)
                except OSError:
                    pass
            
            return {'success': False, 'case_id': case_id}
    
    def add_dl_features_to_radiomics(self, radiomics_df, fold):
        """지정된 fold의 DL embedding 특징을 Radiomics 특징에 추가"""
        if fold not in self.dl_extractors:
            print(f"    경고: Fold {fold}의 DL 추출기를 찾을 수 없음")
            return radiomics_df.drop(columns=['image_path'], errors='ignore')
        
        print(f"    Fold {fold} DL embedding 특징 추가 중...")
        
        combined_features = []
        dl_extractor = self.dl_extractors[fold]
        
        # 각 케이스에 대해 DL embedding 특징 추출 및 결합
        for idx, row in radiomics_df.iterrows():
            case_id = row.get('case_id', idx)
            image_path = row.get('image_path')
            
            if image_path and os.path.exists(image_path):
                try:
                    dl_features = dl_extractor.extract_features_for_case(image_path, case_id)
                    combined_row = row.to_dict()
                    combined_row.update(dl_features)
                    combined_features.append(combined_row)
                    
                except Exception as e:
                    image_filename = os.path.basename(image_path) if image_path else case_id
                    print(f"      케이스 {case_id} ({image_filename}) DL 특징 추출 실패: {e}")
                    combined_features.append(row.to_dict())
            else:
                print(f"      케이스 {case_id}: 이미지 경로 없음 또는 파일 부재")
                combined_features.append(row.to_dict())
        
        result_df = pd.DataFrame(combined_features)
        result_df = result_df.drop(columns=['image_path'], errors='ignore')
        
        # 특징 개수 확인
        dl_feature_count = len([col for col in result_df.columns if col.startswith('dl_embedding_')])
        radiomics_count = len([col for col in result_df.columns 
                             if not col.startswith('dl_embedding_') and col not in ['severity', 'case_id', 'data_source']])
        
        print(f"    Fold {fold}: {radiomics_count}개 radiomics + {dl_feature_count}개 DL embedding 특징")
        
        return result_df
    
    def _print_extraction_summary(self, set_name, processed_cases, skipped_cases):
        """특징 추출 결과 요약 출력"""
        print(f"\n    --- '{set_name}' 세트 특징 추출 요약 ---")
        print(f"    성공적으로 처리된 케이스 수: {len(processed_cases)}")
        
        unique_skipped = sorted(list(set(skipped_cases)))
        if unique_skipped:
            print(f"    건너뛴 고유 케이스 수: {len(unique_skipped)}")

    # 기존 extract_features_for_set 메소드는 하위 호환성을 위해 유지
    def extract_features_for_set(self, image_dir, label_dir, set_name, patient_info_map, mode='binary'):
        """하위 호환성을 위한 기존 인터페이스"""
        return self.extract_radiomics_features_for_set(image_dir, label_dir, set_name, patient_info_map, mode)