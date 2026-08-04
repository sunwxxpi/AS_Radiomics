import os
import shutil
import random
from natsort import natsorted
from tqdm import tqdm

# --- 경로 설정 ---
# 원본 데이터가 위치한 기본 디렉토리
src_base = "/home/psw/AS_Radiomics/data/datasets_raw/KMU/cardiac CT"
# nnUNet 형식으로 정리된 데이터가 저장될 디렉토리
dst_base = "/home/psw/AS_Radiomics/data/datasets/Dataset001_KMU_Cardiac_AVC"

# --- nnUNet 디렉토리 구조 생성 ---
# nnUNet에서 요구하는 표준 폴더 구조 생성
imagesTr_dir = os.path.join(dst_base, "imagesTr")    # 훈련용 이미지
imagesVal_dir = os.path.join(dst_base, "imagesVal")  # 검증용 이미지
labelsTr_dir = os.path.join(dst_base, "labelsTr")    # 훈련용 라벨
labelsVal_dir = os.path.join(dst_base, "labelsVal")  # 검증용 라벨

for d in (imagesTr_dir, imagesVal_dir, labelsTr_dir, labelsVal_dir):
    os.makedirs(d, exist_ok=True)

# --- 환자 디렉토리 탐색 함수 ---
def find_kudh_directories(root_path, max_depth=10):
    """
    재귀적으로 KUDH로 시작하는 환자 디렉토리를 찾는 함수
    
    Args:
        root_path (str): 탐색을 시작할 루트 경로
        max_depth (int): 최대 탐색 깊이 (무한 루프 방지)
    
    Returns:
        list: (환자ID, 환자경로) 튜플의 리스트
    """
    kudh_dirs = []
    
    def recursive_search(current_path, current_depth):
        """재귀적으로 디렉토리를 탐색하는 내부 함수"""
        if current_depth > max_depth:
            return
        
        try:
            entries = os.listdir(current_path)
            for entry in tqdm(entries, desc=f"Searching depth {current_depth}", leave=False):
                entry_path = os.path.join(current_path, entry)
                if not os.path.isdir(entry_path):
                    continue
                
                if entry.startswith("KUDH"):
                    # KUDH로 시작하는 디렉토리 발견
                    kudh_dirs.append((entry, entry_path))
                    tqdm.write(f"Found KUDH directory: {entry}")
                elif entry == "calcium_zero":
                    # calcium_zero 디렉토리는 건너뛰기
                    tqdm.write(f"Skipping calcium_zero directory: {entry_path}")
                    continue
                else:
                    # KUDH가 아닌 폴더는 더 깊이 탐색
                    recursive_search(entry_path, current_depth + 1)
        except PermissionError:
            # 권한이 없는 디렉토리는 건너뛰기
            pass
    
    print("Starting KUDH directory search...")
    recursive_search(root_path, 0)
    return kudh_dirs

# --- FoV 타입 판별 함수 ---
def determine_fov_type(patient_dir):
    """
    환자 디렉토리의 경로에서 FoV 타입을 판별 (Large FoV 또는 Small FoV 폴더)
    
    Args:
        patient_dir (str): 환자 디렉토리 경로
    
    Returns:
        str: 'Large' 또는 'Small' 또는 'Unknown'
    """
    sub_dir = os.path.join(patient_dir, "0002")
    if not os.path.isdir(sub_dir):
        return 'Unknown'
    
    try:
        # 디렉토리 경로에서 FoV 타입 판별
        if 'Large FoV' in patient_dir:
            return 'Large'
        elif 'Small FoV' in patient_dir:
            return 'Small'
        else:
            # 파일명에서도 확인 (백업 방법)
            image_files = [f for f in os.listdir(sub_dir)
                           if f.endswith(".nii.gz") and os.path.isfile(os.path.join(sub_dir, f))]
            
            if not image_files:
                return 'Unknown'
            
            first_file = natsorted(image_files)[0]
            
            if 'Large' in first_file:
                return 'Large'
            elif 'Small' in first_file:
                return 'Small'
            else:
                return 'Unknown'
    except:
        return 'Unknown'

# --- STEP 1: 환자 디렉토리 수집 ---
print("=== STEP 1: 환자 디렉토리 수집 ===")
patient_info = find_kudh_directories(src_base)
# 자연수 기준으로 정렬 (KUDH001, KUDH002, ... 순서)
patient_info = natsorted(patient_info, key=lambda x: x[0])
print(f"총 {len(patient_info)}개의 KUDH 디렉토리를 발견했습니다.")

# --- STEP 2: 데이터 유효성 검사 및 FoV 분류 ---
print("\n=== STEP 2: 데이터 유효성 검사 및 FoV 분류 ===")
log_messages = []  # 로그 메시지 저장용 리스트
valid_patients = []  # 유효한 환자 리스트
large_fov_patients = []  # Large FoV 환자 리스트
small_fov_patients = []  # Small FoV 환자 리스트

for pid, pdir in tqdm(patient_info, desc="Validating patients and determining FoV"):
    # 각 환자의 0002 폴더 존재 여부 확인
    dir0002 = os.path.join(pdir, "0002")
    if not os.path.isdir(dir0002):
        log_messages.append(f"0002 folder not found for patient: {pid}")
        continue

    # FoV 타입 판별
    fov_type = determine_fov_type(pdir)
    if fov_type == 'Unknown':
        log_messages.append(f"FoV type unknown for patient: {pid}")
        continue
    
    # 유효한 환자로 분류
    valid_patients.append((pid, pdir, fov_type))
    
    # FoV 타입별로 분류
    if fov_type == 'Large':
        large_fov_patients.append((pid, pdir, fov_type))
    elif fov_type == 'Small':
        small_fov_patients.append((pid, pdir, fov_type))

print(f"유효한 환자: {len(valid_patients)}명")
print(f"Large FoV 환자: {len(large_fov_patients)}명")
print(f"Small FoV 환자: {len(small_fov_patients)}명")
print(f"유효하지 않은 환자: {len(patient_info) - len(valid_patients)}명")

# --- STEP 3: FoV별 랜덤 분할 후 Train/Validation 합성 (80:20) ---
print("\n=== STEP 3: FoV별 랜덤 분할 후 Train/Validation 합성 ===")

# 랜덤 시드 설정 (재현 가능한 결과를 위해)
random.seed(42)

# Large FoV 환자들을 랜덤하게 섞기
random.shuffle(large_fov_patients)
large_split_index = int(len(large_fov_patients) * 0.8)
large_train = large_fov_patients[:large_split_index]
large_test = large_fov_patients[large_split_index:]

# Small FoV 환자들을 랜덤하게 섞기
random.shuffle(small_fov_patients)
small_split_index = int(len(small_fov_patients) * 0.8)
small_train = small_fov_patients[:small_split_index]
small_test = small_fov_patients[small_split_index:]

# Train/Test 세트 합성
train_patients = large_train + small_train
test_patients = large_test + small_test

# 최종적으로 train과 test 각각 환자번호 순서대로 정렬
train_patients = natsorted(train_patients, key=lambda x: x[0])
test_patients = natsorted(test_patients, key=lambda x: x[0])

print(f"훈련 데이터: {len(train_patients)}명 (Large: {len(large_train)}명, Small: {len(small_train)}명)")
print(f"검증 데이터: {len(test_patients)}명 (Large: {len(large_test)}명, Small: {len(small_test)}명)")

# FoV 비율 확인
large_train_ratio = len(large_train) / len(train_patients) if train_patients else 0
small_train_ratio = len(small_train) / len(train_patients) if train_patients else 0
large_test_ratio = len(large_test) / len(test_patients) if test_patients else 0
small_test_ratio = len(small_test) / len(test_patients) if test_patients else 0

print(f"훈련 데이터 FoV 비율 - Large: {large_train_ratio:.2%}, Small: {small_train_ratio:.2%}")
print(f"검증 데이터 FoV 비율 - Large: {large_test_ratio:.2%}, Small: {small_test_ratio:.2%}")

# 전역 성공 처리 카운터
global_success_count = 0

# --- STEP 4: 파일 복사 함수 정의 ---
def copy_patient_files(patient_list, images_target, labels_target, start_count=0):
    """
    환자 파일들을 nnUNet 형식으로 복사하는 함수
    
    Args:
        patient_list: 처리할 환자 리스트 (pid, pdir, fov_type 튜플)
        images_target: 이미지 파일들이 저장될 대상 디렉토리
        labels_target: 라벨 파일들이 저장될 대상 디렉토리
        start_count: 시작 카운트 번호
    
    Returns:
        int: 다음 카운트 번호
    """
    global global_success_count
    count = start_count
    
    for pid, pdir, _ in tqdm(patient_list, desc="Copying files"):
        sub_dir = os.path.join(pdir, "0002")
        
        # 1) 이미지 파일 처리
        # 0002 폴더에서 .nii.gz 파일 찾기
        image_files = [f for f in os.listdir(sub_dir)
                       if f.endswith(".nii.gz") and os.path.isfile(os.path.join(sub_dir, f))]
        
        if not image_files:
            log_messages.append(f"Image file not found for patient: {pid}")
            continue
            
        # 자연수 정렬 후 첫 번째 파일 선택
        image_files = natsorted(image_files)
        image_src = os.path.join(sub_dir, image_files[0])

        # 2) Ground Truth 파일 처리
        # results 폴더에서 _000.nii.gz로 끝나는 파일 찾기
        results_dir = os.path.join(sub_dir, "results")
        if not os.path.isdir(results_dir):
            log_messages.append(f"Results directory not found for patient: {pid}")
            continue
            
        gt_files = [f for f in os.listdir(results_dir) if f.endswith("_000.nii.gz")]
        if not gt_files:
            log_messages.append(f"GT file not found for patient: {pid}")
            continue
            
        # 자연수 정렬 후 첫 번째 파일 선택
        gt_files = natsorted(gt_files)
        gt_src = os.path.join(results_dir, gt_files[0])

        # 3) nnUNet 명명 규칙에 따른 파일 복사
        count += 1
        global_success_count += 1
        
        # nnUNet 표준 파일명: {CaseID}_{SequenceID}_0000.nii.gz (이미지)
        # nnUNet 표준 파일명: {CaseID}_{SequenceID}.nii.gz (라벨)
        new_image_name = f"{pid}_{count:04d}_0000.nii.gz"
        new_label_name = f"{pid}_{count:04d}.nii.gz"
        
        # 실제 파일 복사 수행
        shutil.copy(image_src, os.path.join(images_target, new_image_name))
        shutil.copy(gt_src, os.path.join(labels_target, new_label_name))
    
    return count

# --- STEP 5: 실제 파일 복사 수행 ---
print("\n=== STEP 5: 파일 복사 수행 ===")
print("훈련 데이터 복사 중...")
copy_patient_files(train_patients, imagesTr_dir, labelsTr_dir, start_count=0)

print("검증 데이터 복사 중...")
copy_patient_files(test_patients, imagesVal_dir, labelsVal_dir, start_count=0)

# --- STEP 6: 처리 결과 로그 생성 ---
print("\n=== STEP 6: 처리 결과 로그 생성 ===")

# 통계 계산
total_cases = len(patient_info)                    # 발견된 전체 환자 수
tr_count = len(train_patients)                     # 훈련 분할 환자 수
val_count = len(test_patients)                     # 검증 분할 환자 수
fail_count = total_cases - global_success_count    # 실패한 환자 수

# 로그 메시지 작성
log_messages.append("Organizing KMU Cardiac_AVC dataset in nnUNet format with stratified FoV-balanced split is complete.")
log_messages.append("\n=== Processing Summary ===")
log_messages.append(f"Total cases discovered: {total_cases}\n")
log_messages.append(f"Large FoV cases: {len(large_fov_patients)}")
log_messages.append(f"Small FoV cases: {len(small_fov_patients)}\n")
log_messages.append(f"Training cases: {tr_count} (Large: {len(large_train)}, Small: {len(small_train)})")
log_messages.append(f"Validation cases: {val_count} (Large: {len(large_test)}, Small: {len(small_test)})\n")
log_messages.append(f"Training FoV ratio - Large: {large_train_ratio:.2%}, Small: {small_train_ratio:.2%}")
log_messages.append(f"Validation FoV ratio - Large: {large_test_ratio:.2%}, Small: {small_test_ratio:.2%}\n")
log_messages.append(f"Successfully processed: {global_success_count}")
log_messages.append(f"Failed cases: {fail_count}")

# 로그 파일 저장
results_file = os.path.join(dst_base, "cardiac_avc_nnUNet_results.txt")
with open(results_file, "w", encoding="utf-8") as f:
    for msg in log_messages:
        f.write(msg + "\n")

print(f"처리 완료! 결과 로그: {results_file}")
print(f"성공: {global_success_count}명, 실패: {fail_count}명")
print(f"FoV 균형 분할 완료 - 훈련/검증 모두에서 Large/Small FoV 비율이 유지됩니다.")