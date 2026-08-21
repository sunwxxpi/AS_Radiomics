import os
import re
import logging
from collections import Counter
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import binary_dilation, generate_binary_structure
from radiomics import featureextractor
from trainer.dl_embedding_extractor import DLEmbeddingExtractor

class RadiomicsExtractor:
    """Radiomics와 DL embedding 특징을 추출하는 통합 클래스

    Radiomics 는 케이스당 한 번만 뽑는다. DL embedding 은 development 행이면 자기를 검증으로 뺀 fold 모델, test 행이면 refit 모델로 붙인다.
    """
    
    def __init__(self, geometry_tolerance=1e-5, enable_dl_embedding=False, dl_model_paths=None, dl_model_type='custom', dl_nnunet_config=None,
                 resampled_spacing=None, resample_interpolator='sitkLinear', enable_dilation=False, dilation_iterations=1,
                 degenerate_dilation_iterations=0):
        self.extractor = featureextractor.RadiomicsFeatureExtractor()
        self.extractor.settings['geometryTolerance'] = geometry_tolerance
        # resampled_spacing 은 [x, y, z] mm. None 이면 PyRadiomics 가 리샘플링 단계를 건너뛴다.
        self.extractor.settings['resampledPixelSpacing'] = resampled_spacing
        self.extractor.settings['interpolator'] = resample_interpolator
        self.enable_dl_embedding = enable_dl_embedding
        self.dl_extractors = {}  # fold별 DL embedding 추출기 저장소
        self.enable_dilation = enable_dilation
        self.dilation_iterations = dilation_iterations
        self.degenerate_dilation_iterations = degenerate_dilation_iterations
        self.dilation_rescued_cases = []
        
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
                        raise RuntimeError(f"DL embedding 추출기 초기화 실패 - Fold {fold} ({model_path}): {e}") from e
                else:
                    print(f"  경고: Fold {fold} 모델 파일을 찾을 수 없음: {model_path}\n")
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Radiomics 로거 설정"""
        logger = logging.getLogger("radiomics")
        logger.setLevel(logging.ERROR)
    
    def _apply_dilation(self, label_path, iterations=None):
        """레이블 마스크에 dilation 적용

        실패하면 원본 경로를 그대로 돌려주므로 호출부는 반환값이 인자와 같은지로 성공을 판별한다.
        """
        iterations = self.dilation_iterations if iterations is None else iterations
        try:
            original_img = nib.load(label_path)
            original_data = original_img.get_fdata().astype(np.uint8)
            
            structure = generate_binary_structure(rank=3, connectivity=3)
            dilated_data = binary_dilation(
                input=original_data,
                structure=structure,
                iterations=iterations
            ).astype(original_data.dtype)
            
            dilated_img = nib.Nifti1Image(
                dataobj=dilated_data,
                affine=original_img.affine,
                header=original_img.header
            )
            
            temp_label_path = label_path.replace('.nii.gz', f'_dilated_{iterations}iter_temp.nii.gz')
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
        rescued_before = len(self.dilation_rescued_cases)

        for image_filename in image_files:
            result = self._extract_radiomics_only(
                image_filename, image_dir, label_dir, patient_info_map, set_name)
            
            if result['success']:
                features_list.append(result['features'])
                processed_cases.append(result['case_id'])
            else:
                skipped_cases.append(result['case_id'])
        
        self._print_extraction_summary(set_name, processed_cases, skipped_cases,
                                       self.dilation_rescued_cases[rescued_before:])
        
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
        rescued_by_dilation = False

        if self.enable_dilation:
            final_label_path = self._apply_dilation(label_path)
            if final_label_path != label_path:
                temp_label_path = final_label_path
        
        # Radiomics 특징 추출
        try:
            try:
                result = self.extractor.execute(image_path, final_label_path, label=1)
            except Exception as first_error:
                # 리샘플링 후 선·점으로 축퇴한 마스크만 되살린다. 전역 dilation 은 멀쩡한 케이스의 특징까지 바꾼다.
                if temp_label_path is not None or self.degenerate_dilation_iterations <= 0:
                    raise
                print(f"      추출 실패 ({first_error})")
                print(f"      마스크를 {self.degenerate_dilation_iterations}회 팽창시켜 재시도")
                final_label_path = self._apply_dilation(label_path, self.degenerate_dilation_iterations)
                if final_label_path == label_path:
                    raise
                temp_label_path = final_label_path
                result = self.extractor.execute(image_path, final_label_path, label=1)
                rescued_by_dilation = True

            features = {key: val for key, val in result.items() if not key.startswith('diagnostics_')}
            
            features['case_id'] = case_id
            features['severity'] = label
            features['image_path'] = image_path  # DL 특징 추출용 경로 저장
            # 데이터 소스 정보 추가 (디렉토리 기반 분할을 위해)
            data_source = 'train' if 'Tr' in set_name else 'val'
            features['data_source'] = data_source
            
            radiomics_count = len([k for k in features.keys() 
                                 if k not in ['case_id', 'severity', 'image_path', 'data_source']])
            
            if rescued_by_dilation:
                self.dilation_rescued_cases.append(case_id)
                print(f"      성공: {radiomics_count}개 radiomics 특징 추출 "
                      f"(마스크 {self.degenerate_dilation_iterations}회 팽창본)")
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
            print(f"      오류: Radiomics 특징 추출 실패 - {e}")
            
            # 임시 파일 정리
            if temp_label_path and os.path.exists(temp_label_path):
                try:
                    os.remove(temp_label_path)
                except OSError:
                    pass
            
            return {'success': False, 'case_id': case_id}
    
    def add_oof_dl_features_to_radiomics(self, radiomics_df, fold_assignment_path):
        """development 행은 그 케이스를 검증으로 뺀 fold 모델로, test 행은 refit 모델로 DL embedding 을 붙인다.

        development 행에 그 행을 학습에 쓴 모델의 embedding 을 주면 융합 분류기가 test 에는 없을 과적합된 표현 위에서 학습된다.
        모든 행의 출처를 추출 전에 정하고, 한 행이라도 비거나 차원이 어긋나면 프레임을 만들지 않고 멈춘다.
        """
        if not os.path.exists(fold_assignment_path):
            raise FileNotFoundError(f"fold 배정 파일이 없다: {fold_assignment_path} — DL 5-fold 를 먼저 돌린다")

        assignment = pd.read_csv(fold_assignment_path)
        fold_of_case = dict(zip(assignment['case_id'], assignment['fold'].astype(int)))
        source_of_case = self._resolve_embedding_sources(radiomics_df, fold_of_case)

        counts = Counter(source_of_case.values())
        print("    DL embedding 출처: " + ", ".join(f"{source}={counts[source]}건" for source in sorted(counts, key=str)))

        combined_features = []
        embedding_dim = None

        for _, row in radiomics_df.iterrows():
            case_id = row['case_id']
            source = source_of_case[case_id]

            dl_features = self.dl_extractors[source].extract_features_for_case(row['image_path'], case_id)
            if not dl_features:
                raise RuntimeError(f"케이스 {case_id}: {source} 모델의 DL embedding 이 비었다")
            if embedding_dim is None:
                embedding_dim = len(dl_features)
            elif len(dl_features) != embedding_dim:
                raise RuntimeError(f"케이스 {case_id}: DL embedding 이 {len(dl_features)}차원, 앞선 행은 {embedding_dim}차원")

            combined_row = row.to_dict()
            combined_row.update(dl_features)
            combined_features.append(combined_row)

        result_df = pd.DataFrame(combined_features).drop(columns=['image_path'], errors='ignore')
        radiomics_count = len([col for col in result_df.columns
                               if not col.startswith('dl_embedding_') and col not in ['severity', 'case_id', 'data_source']])
        print(f"    OOF: {radiomics_count}개 radiomics + {embedding_dim}개 DL embedding 특징 ({len(result_df)}건)")

        return result_df

    def _resolve_embedding_sources(self, radiomics_df, fold_of_case):
        """행마다 어느 모델로 embedding 을 뽑을지 `case_id → fold 번호 | 'refit'` 으로 정한다.

        추출은 케이스당 수 초라 다 돌린 뒤 실패하면 그 시간을 버린다. 그래서 여기서 한 번에 검사하고 문제를 모아 던진다.
        """
        source_of_case = {}
        missing_image, missing_assignment, leaked_test, missing_model = [], [], [], []

        for _, row in radiomics_df.iterrows():
            case_id = row['case_id']
            image_path = row.get('image_path')

            if not image_path or not os.path.exists(image_path):
                missing_image.append(case_id)
                continue

            if row['data_source'] == 'train':
                if case_id not in fold_of_case:
                    missing_assignment.append(case_id)
                    continue
                source = fold_of_case[case_id]
            else:
                if case_id in fold_of_case:
                    leaked_test.append(case_id)
                    continue
                source = 'refit'

            if source not in self.dl_extractors:
                missing_model.append(f"{case_id}({source})")
                continue

            source_of_case[case_id] = source

        problems = []
        if missing_image:
            problems.append(f"이미지 부재 {len(missing_image)}건 {missing_image[:5]}")
        if missing_assignment:
            problems.append(f"fold 배정에 없는 development 케이스 {len(missing_assignment)}건 {missing_assignment[:5]}")
        if leaked_test:
            problems.append(f"fold 배정에 들어 있는 test 케이스 {len(leaked_test)}건 {leaked_test[:5]}"
                            " — DL 학습이 test 를 봤거나 배정 파일이 다른 분할의 것이다")
        if missing_model:
            problems.append(f"모델이 없는 케이스 {len(missing_model)}건 {missing_model[:5]}")
        if problems:
            raise RuntimeError("OOF DL embedding 을 만들 수 없다: " + " / ".join(problems))

        return source_of_case

    def _print_extraction_summary(self, set_name, processed_cases, skipped_cases, rescued_cases=()):
        """특징 추출 결과 요약 출력"""
        print(f"\n    --- '{set_name}' 세트 특징 추출 요약 ---")
        print(f"    성공적으로 처리된 케이스 수: {len(processed_cases)}")
        
        unique_skipped = sorted(list(set(skipped_cases)))
        if unique_skipped:
            print(f"    건너뛴 고유 케이스 수: {len(unique_skipped)}")

        if rescued_cases:
            print(f"    마스크 팽창으로 되살린 케이스 수: {len(rescued_cases)}")
            for case_id in sorted(rescued_cases):
                print(f"      - {case_id}")

    # 기존 extract_features_for_set 메소드는 하위 호환성을 위해 유지
    def extract_features_for_set(self, image_dir, label_dir, set_name, patient_info_map, mode='binary'):
        """하위 호환성을 위한 기존 인터페이스"""
        return self.extract_radiomics_features_for_set(image_dir, label_dir, set_name, patient_info_map, mode)