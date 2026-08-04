import os
import shutil
import argparse
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split


def load_severity_mapping(csv_path, mode='multi'):
    """CSV에서 환자별 중증도 정보 로딩"""
    print(f"CSV 파일 로딩: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 필수 컬럼 확인
    required_columns = ['1차년도연구번호', 'AV_binaryclassification', 'AS ']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV 파일에 필수 컬럼이 없습니다: {required_columns}")
    
    # AS 컬럼명 변경 및 값 정규화
    df.rename(columns={'AS ': 'AS_grade'}, inplace=True)
    df['AS_grade'] = df['AS_grade'].astype(str).str.strip().str.lower()
    df['AS_grade'] = df['AS_grade'].replace('nan', 'unknown')
    df['AS_grade'] = df['AS_grade'].fillna('unknown')
    
    if mode == 'binary':
        # Binary 모드: AV_binaryclassification 사용
        binary_df = df.dropna(subset=['AV_binaryclassification']).copy()
        severity_map = binary_df.set_index('1차년도연구번호')['AV_binaryclassification'].to_dict()
        print(f"Binary 모드 - 총 {len(severity_map)}개 환자")
        print(f"클래스 분포: {binary_df['AV_binaryclassification'].value_counts().to_dict()}")
    
    else:
        # Multi 모드: AS_grade를 3-class로 변환
        def map_to_three_class(grade):
            grade = str(grade).strip().lower()
            if grade in ['none', 'no']:
                return 'normal'
            elif grade in ['mild', 'moderate', 'pseudosevere']:
                return 'nonsevere'
            elif grade in ['severe', 'very severe']:
                return 'severe'
            else:
                return 'unknown'
        
        df['AS_grade_3class'] = df['AS_grade'].apply(map_to_three_class)
        multi_df = df[df['AS_grade_3class'] != 'unknown']
        severity_map = multi_df.set_index('1차년도연구번호')['AS_grade_3class'].to_dict()
        print(f"Multi 모드 - 총 {len(severity_map)}개 환자")
        print(f"클래스 분포: {multi_df['AS_grade_3class'].value_counts().to_dict()}")
    
    return severity_map


def extract_patient_id(filename):
    """파일명에서 환자 ID 추출"""
    # KUDH0001_0001_0000.nii.gz -> KUDH0001
    parts = filename.split('_')
    if len(parts) >= 2:
        return parts[0]
    return None


def collect_files_with_severity(source_dir, severity_map):
    """단일 소스 디렉토리에서 파일들을 수집하고 중증도별로 분류"""
    files_by_severity = defaultdict(list)
    missing_patients = set()
    
    print(f"\n소스 디렉토리 검사: {source_dir}")
    
    # 이미지 디렉토리 찾기
    img_dir = os.path.join(source_dir, 'imagesVal')
    label_dir = os.path.join(source_dir, 'predVal')
    
    if not os.path.exists(img_dir) or not os.path.exists(label_dir):
        raise ValueError(f"디렉토리가 존재하지 않습니다 - {img_dir} 또는 {label_dir}")
    
    # 이미지 파일들 수집
    for img_file in os.listdir(img_dir):
        if not img_file.endswith('.nii.gz'):
            continue
            
        patient_id = extract_patient_id(img_file)
        if not patient_id:
            continue
        
        # 해당 레이블 파일 찾기
        label_file = img_file.replace('_0000.nii.gz', '.nii.gz')
        label_path = os.path.join(label_dir, label_file)
        
        if not os.path.exists(label_path):
            print(f"경고: 레이블 파일 없음 - {label_file}")
            continue
        
        # 중증도 정보 확인
        if patient_id not in severity_map:
            missing_patients.add(patient_id)
            continue
        
        severity = severity_map[patient_id]
        img_path = os.path.join(img_dir, img_file)
        
        files_by_severity[severity].append({
            'patient_id': patient_id,
            'img_file': img_file,
            'label_file': label_file,
            'img_path': img_path,
            'label_path': label_path
        })
    
    if missing_patients:
        print(f"\n경고: CSV에서 중증도 정보를 찾을 수 없는 환자 {len(missing_patients)}명:")
        print(sorted(list(missing_patients))[:10], "..." if len(missing_patients) > 10 else "")
    
    return files_by_severity


def perform_stratified_split(files_by_severity, test_size=0.2, random_state=42):
    """환자 단위로 중증도별 stratified split 수행"""
    print(f"\n환자 단위 Stratified split 수행 (test_size={test_size}, random_state={random_state})")
    
    # 먼저 모든 환자의 중증도 정보 수집
    patient_severity = {}
    for severity, files in files_by_severity.items():
        for file_info in files:
            patient_id = file_info['patient_id']
            if patient_id in patient_severity and patient_severity[patient_id] != severity:
                print(f"경고: 환자 {patient_id}에 대해 중증도가 일치하지 않음: {patient_severity[patient_id]} vs {severity}")
            patient_severity[patient_id] = severity
    
    # 중증도별로 환자 그룹화
    patients_by_severity = defaultdict(list)
    for patient_id, severity in patient_severity.items():
        patients_by_severity[severity].append(patient_id)
    
    # 중증도별로 환자 단위 split
    train_patients = set()
    val_patients = set()
    
    for severity, patients in patients_by_severity.items():
        print(f"\n{severity} 클래스: {len(patients)}명 환자")
        
        if len(patients) < 2:
            print(f"경고: {severity} 클래스에 환자가 {len(patients)}명만 있어 분할 불가")
            train_patients.update(patients)
            continue
        
        # 환자 단위로 stratified split
        train_pts, val_pts = train_test_split(
            patients, 
            test_size=test_size, 
            random_state=random_state,
            stratify=None  # 환자 수가 적을 수 있으므로
        )
        
        train_patients.update(train_pts)
        val_patients.update(val_pts)
        
        print(f"  -> 학습: {len(train_pts)}명 환자")
        print(f"  -> 검증: {len(val_pts)}명 환자")
    
    # 환자 단위 split에 따라 파일들 배정
    train_files = []
    val_files = []
    
    for severity, files in files_by_severity.items():
        for file_info in files:
            if file_info['patient_id'] in train_patients:
                train_files.append(file_info)
            elif file_info['patient_id'] in val_patients:
                val_files.append(file_info)
    
    print(f"\n=== 최종 파일 배정 ===")
    print(f"학습 파일: {len(train_files)}개")
    print(f"검증 파일: {len(val_files)}개")
    
    return train_files, val_files


def generate_nnunet_filename(patient_id, case_id, is_train=True):
    """nnUNet 형식에 맞는 파일명 생성"""
    # Train과 Val 모두 동일한 형식: 환자번호_AAAA_0000.nii.gz
    return f"{patient_id}_{case_id:04d}_0000.nii.gz"


def copy_files_to_dataset(train_files, val_files, output_dir):
    """파일들을 Dataset 구조로 복사"""
    print(f"\nDataset 디렉토리 생성: {output_dir}")
    
    # 디렉토리 생성
    dirs_to_create = [
        os.path.join(output_dir, 'imagesTr'),
        os.path.join(output_dir, 'imagesVal'),
        os.path.join(output_dir, 'labelsTr'),
        os.path.join(output_dir, 'labelsVal')
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
    
    # 환자 ID 기준으로 정렬된 Train 파일 복사 (0001부터 순차 번호)
    print(f"학습 파일 복사: {len(train_files)}개")
    # 환자 ID로 정렬
    train_files_sorted = sorted(train_files, key=lambda x: x['patient_id'])
    for i, file_info in enumerate(train_files_sorted):
        src_img = file_info['img_path']
        src_label = file_info['label_path']
        
        # nnUNet 형식으로 파일명 생성
        case_id = i + 1  # 1부터 시작
        new_img_filename = generate_nnunet_filename(file_info['patient_id'], case_id, is_train=True)
        new_label_filename = new_img_filename.replace('_0000.nii.gz', '.nii.gz')
        
        dst_img = os.path.join(output_dir, 'imagesTr', new_img_filename)
        dst_label = os.path.join(output_dir, 'labelsTr', new_label_filename)
        
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_label, dst_label)
        
        # 파일명 매핑 정보 저장
        file_info['new_img_file'] = new_img_filename
        file_info['new_label_file'] = new_label_filename
    
    # 환자 ID 기준으로 정렬된 Val 파일 복사 (0001부터 순차 번호)
    print(f"검증 파일 복사: {len(val_files)}개")
    # 환자 ID로 정렬
    val_files_sorted = sorted(val_files, key=lambda x: x['patient_id'])
    for i, file_info in enumerate(val_files_sorted):
        src_img = file_info['img_path']
        src_label = file_info['label_path']
        
        # nnUNet 형식으로 파일명 생성
        case_id = i + 1  # 1부터 시작
        new_img_filename = generate_nnunet_filename(file_info['patient_id'], case_id, is_train=False)
        new_label_filename = new_img_filename.replace('_0000.nii.gz', '.nii.gz')  # Label은 _0000 제거
        
        dst_img = os.path.join(output_dir, 'imagesVal', new_img_filename)
        dst_label = os.path.join(output_dir, 'labelsVal', new_label_filename)
        
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_label, dst_label)
        
        # 파일명 매핑 정보 저장
        file_info['new_img_file'] = new_img_filename
        file_info['new_label_file'] = new_label_filename
        
    # 정렬된 리스트를 원본 리스트에 업데이트
    train_files[:] = train_files_sorted
    val_files[:] = val_files_sorted


def save_split_info(train_files, val_files, severity_map, output_dir, dataset_name):
    """분할 정보를 CSV로 저장"""
    print(f"\n분할 정보 저장")
    
    # 분할 결과 정리
    split_info = []
    
    for file_info in train_files:
        split_info.append({
            'patient_id': file_info['patient_id'],
            'original_img_file': file_info['img_file'],
            'original_label_file': file_info['label_file'],
            'new_img_file': file_info.get('new_img_file', file_info['img_file']),
            'new_label_file': file_info.get('new_label_file', file_info['label_file']),
            'severity': severity_map[file_info['patient_id']],
            'split': 'train'
        })
    
    for file_info in val_files:
        split_info.append({
            'patient_id': file_info['patient_id'],
            'original_img_file': file_info['img_file'],
            'original_label_file': file_info['label_file'],
            'new_img_file': file_info.get('new_img_file', file_info['img_file']),
            'new_label_file': file_info.get('new_label_file', file_info['label_file']),
            'severity': severity_map[file_info['patient_id']],
            'split': 'val'
        })
    
    # DataFrame으로 변환 및 저장
    df = pd.DataFrame(split_info)
    csv_path = os.path.join(output_dir, f'{dataset_name}_info.csv')
    df.to_csv(csv_path, index=False)
    print(f"분할 정보 저장: {csv_path}")
    
    # 분할 통계 출력
    print(f"\n=== 최종 분할 통계 ===")
    for split in ['train', 'val']:
        split_df = df[df['split'] == split]
        print(f"\n{split.upper()} 세트:")
        print(f"  총 파일 수: {len(split_df)}")
        severity_counts = split_df['severity'].value_counts()
        for severity, count in severity_counts.items():
            print(f"  {severity}: {count}개")


def main():
    # 데이터셋 타입별 설정 정의
    DATASET_CONFIGS = {
        'total': {
            'source_dir': '/home/psw/AS_Radiomics/data/datasets/Dataset001_KMU_Cardiac_AVC_TOTAL',
            'output_dir': './Dataset003_total',
            'dataset_name': 'Dataset003_total'
        },
        'total_cropped': {
            'source_dir': '/home/psw/AS_Radiomics/data/datasets/Dataset001_KMU_Cardiac_AVC_TOTAL_CROPPED',
            'output_dir': 'Dataset003_total_cropped',
            'dataset_name': 'Dataset003_total_cropped'
        }
    }
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', 
                       default='/home/psw/AS_Radiomics/data/AS_CRF.csv')
    parser.add_argument('--mode', choices=['binary', 'multi'], default='multi')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--dataset_type', choices=['total', 'total_cropped'], default='total')
    
    args = parser.parse_args()
    
    # 선택된 데이터셋 타입에 따라 경로들 설정
    config = DATASET_CONFIGS[args.dataset_type]
    args.source_dir = config['source_dir']
    args.output_dir = config['output_dir'] 
    args.dataset_name = config['dataset_name']
    
    print(f"=== {args.dataset_name.upper()} 생성 시작 ===")
    print(f"CSV 파일: {args.csv_path}")
    print(f"분류 모드: {args.mode}")
    print(f"검증 보보 비율: {args.test_size}")
    print(f"랜덤 시드: {args.random_state}")
    print(f"소스 디렉토리: {args.source_dir}")
    print(f"출력 디렉토리: {args.output_dir}")
    
    # 1. 중증도 정보 로딩
    severity_map = load_severity_mapping(args.csv_path, args.mode)
    
    # 2. 파일 수집 및 중증도별 분류
    files_by_severity = collect_files_with_severity(args.source_dir, severity_map)
    
    # 3. Stratified split 수행
    train_files, val_files = perform_stratified_split(
        files_by_severity, 
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # 4. Dataset 디렉토리 생성 및 파일 복사
    copy_files_to_dataset(train_files, val_files, args.output_dir)
    
    # 5. 분할 정보 저장
    save_split_info(train_files, val_files, severity_map, args.output_dir, args.dataset_name)
    
    print(f"\n=== {args.dataset_name.upper()} 생성 완료 ===")
    print(f"출력 디렉토리: {args.output_dir}")


if __name__ == '__main__':
    main()