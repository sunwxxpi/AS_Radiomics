import os
import shutil
from natsort import natsorted
from tqdm import tqdm

# --- 경로 설정 ---
# 원본 데이터가 위치한 기본 디렉토리
src_base = "/home/psw/AS_Radiomics/data/datasets_raw/KMU/cardiac CT"
# 누락 데이터가 위치한 디렉토리
missing_base = "/home/psw/AS_Radiomics/data/datasets_raw/KMU/cardiac CT/cardiac_ct_1차_누락"
# nnUNet 형식으로 정리된 데이터가 저장될 디렉토리
dst_base = "/home/psw/AS_Radiomics/data/datasets/Dataset001_KMU_Cardiac_AVC_TOTAL"

# --- nnUNet 디렉토리 구조 생성 ---
# test 비율 1.0이므로 imagesVal만 사용
imagesVal_dir = os.path.join(dst_base, "imagesVal")    # 테스트용 이미지

os.makedirs(imagesVal_dir, exist_ok=True)

# --- 환자 디렉토리 탐색 함수 ---
def find_kudh_directories(root_path, max_depth=5):
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
    
    print(f"Starting KUDH directory search in: {root_path}")
    recursive_search(root_path, 0)
    return kudh_dirs

# --- 누락 디렉토리에서 KUDH 파일 탐색 함수 ---
def find_kudh_files_in_missing_dir(missing_dir_path):
    """
    누락 디렉토리에서 KUDH로 시작하는 .nii.gz 파일들을 찾는 함수
    (Large FoV, Small FoV 폴더에서 직접 파일을 찾음)
    
    Args:
        missing_dir_path (str): 누락 디렉토리 경로
    
    Returns:
        list: (환자ID, 파일경로, FoV타입) 튜플의 리스트
    """
    kudh_files = []
    
    # Large FoV 폴더 확인
    large_fov_dir = os.path.join(missing_dir_path, "Large FoV")
    if os.path.isdir(large_fov_dir):
        try:
            files = os.listdir(large_fov_dir)
            for file in files:
                if file.startswith("KUDH") and file.endswith(".nii.gz"):
                    patient_id = file.replace(".nii.gz", "")
                    file_path = os.path.join(large_fov_dir, file)
                    kudh_files.append((patient_id, file_path, "Large"))
                    print(f"Found KUDH file in Large FoV: {patient_id}")
        except PermissionError:
            pass
    
    # Small FoV 폴더 확인
    small_fov_dir = os.path.join(missing_dir_path, "Small FoV")
    if os.path.isdir(small_fov_dir):
        try:
            files = os.listdir(small_fov_dir)
            for file in files:
                if file.startswith("KUDH") and file.endswith(".nii.gz"):
                    patient_id = file.replace(".nii.gz", "")
                    file_path = os.path.join(small_fov_dir, file)
                    kudh_files.append((patient_id, file_path, "Small"))
                    print(f"Found KUDH file in Small FoV: {patient_id}")
        except PermissionError:
            pass
    
    return kudh_files

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

# --- GT 파일 존재 여부 확인 함수 ---
def has_gt_file(patient_dir):
    """
    환자 디렉토리에 GT 파일이 있는지 확인
    
    Args:
        patient_dir (str): 환자 디렉토리 경로
    
    Returns:
        bool: GT 파일 존재 여부
    """
    sub_dir = os.path.join(patient_dir, "0002")
    if not os.path.isdir(sub_dir):
        return False
    
    results_dir = os.path.join(sub_dir, "results")
    if not os.path.isdir(results_dir):
        return False
    
    gt_files = [f for f in os.listdir(results_dir) if f.endswith("_000.nii.gz")]
    return len(gt_files) > 0

# --- STEP 1: 모든 환자 디렉토리 수집 ---
print("=== STEP 1: 모든 환자 디렉토리 수집 ===")

# 기본 디렉토리에서 수집
print("기본 디렉토리에서 환자 수집...")
patient_info_base = find_kudh_directories(src_base)

# 누락 디렉토리에서 수집 (구조가 다르므로 별도 함수 사용)
print("누락 디렉토리에서 환자 수집...")
patient_files_missing = find_kudh_files_in_missing_dir(missing_base)

# 기본 디렉토리 환자 정보 (디렉토리 형태)
all_patient_info = patient_info_base

# 자연수 기준으로 정렬 (KUDH001, KUDH002, ... 순서)
all_patient_info = natsorted(all_patient_info, key=lambda x: x[0])
patient_files_missing = natsorted(patient_files_missing, key=lambda x: x[0])

print(f"총 {len(all_patient_info)}개의 KUDH 디렉토리를 발견했습니다.")
print(f"- 기본 디렉토리: {len(patient_info_base)}개")
print(f"총 {len(patient_files_missing)}개의 KUDH 누락 파일을 발견했습니다.")

# --- STEP 2: 데이터 유효성 검사 및 분류 ---
print("\n=== STEP 2: 데이터 유효성 검사 및 분류 ===")
log_messages = []  # 로그 메시지 저장용 리스트
valid_patients_with_gt = []    # GT가 있는 유효한 환자 리스트
valid_patients_without_gt = [] # GT가 없는 유효한 환자 리스트
large_fov_with_gt = []         # GT가 있는 Large FoV 환자
small_fov_with_gt = []         # GT가 있는 Small FoV 환자
large_fov_without_gt = []      # GT가 없는 Large FoV 환자
small_fov_without_gt = []      # GT가 없는 Small FoV 환자

# 1) 기본 디렉토리의 환자들 처리
for pid, pdir in tqdm(all_patient_info, desc="Validating base directory patients"):
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
    
    # GT 파일 존재 여부 확인
    has_gt = has_gt_file(pdir)
    
    # 분류
    if has_gt:
        valid_patients_with_gt.append((pid, pdir, fov_type))
        if fov_type == 'Large':
            large_fov_with_gt.append((pid, pdir, fov_type))
        elif fov_type == 'Small':
            small_fov_with_gt.append((pid, pdir, fov_type))
    else:
        valid_patients_without_gt.append((pid, pdir, fov_type))
        if fov_type == 'Large':
            large_fov_without_gt.append((pid, pdir, fov_type))
        elif fov_type == 'Small':
            small_fov_without_gt.append((pid, pdir, fov_type))

# 2) 누락 디렉토리의 파일들 처리 (GT가 없는 것으로 간주)
for pid, file_path, fov_type in tqdm(patient_files_missing, desc="Processing missing directory files"):
    # 누락 디렉토리의 파일들은 GT가 없는 것으로 간주
    valid_patients_without_gt.append((pid, file_path, fov_type))
    if fov_type == 'Large':
        large_fov_without_gt.append((pid, file_path, fov_type))
    elif fov_type == 'Small':
        small_fov_without_gt.append((pid, file_path, fov_type))

total_valid = len(valid_patients_with_gt) + len(valid_patients_without_gt)
print(f"유효한 환자: {total_valid}명")
print(f"- GT가 있는 환자: {len(valid_patients_with_gt)}명 (Large: {len(large_fov_with_gt)}명, Small: {len(small_fov_with_gt)}명)")
print(f"- GT가 없는 환자: {len(valid_patients_without_gt)}명 (Large: {len(large_fov_without_gt)}명, Small: {len(small_fov_without_gt)}명)")
print(f"유효하지 않은 환자: {len(all_patient_info) - total_valid}명")

# --- STEP 3: 모든 환자를 테스트 세트에 포함 (test 비율 1.0) ---
print("\n=== STEP 3: 모든 환자를 테스트 세트에 포함 ===")

# GT가 있는 환자와 없는 환자를 모두 합치기
all_valid_patients = valid_patients_with_gt + valid_patients_without_gt

# 환자번호 순서대로 정렬
all_valid_patients = natsorted(all_valid_patients, key=lambda x: x[0])

print(f"테스트 데이터: {len(all_valid_patients)}명")
print(f"- GT가 있는 환자: {len(valid_patients_with_gt)}명")
print(f"- GT가 없는 환자: {len(valid_patients_without_gt)}명")

# 전역 성공 처리 카운터
global_success_count = 0

# --- STEP 4: 파일 복사 함수 정의 ---
def copy_patient_files_total(patient_list, images_target, start_count=0):
    """
    모든 환자 파일들을 nnUNet 형식으로 복사하는 함수 (이미지만 복사)
    
    Args:
        patient_list: 처리할 환자 리스트 (pid, path_or_dir, fov_type 튜플)
                      - 기본 디렉토리: path_or_dir은 환자 디렉토리 경로
                      - 누락 디렉토리: path_or_dir은 .nii.gz 파일 경로
        images_target: 이미지 파일들이 저장될 대상 디렉토리
        start_count: 시작 카운트 번호
    
    Returns:
        int: 다음 카운트 번호
    """
    global global_success_count
    count = start_count
    
    for pid, path_or_dir, _ in tqdm(patient_list, desc="Copying files"):
        image_src = None
        
        # 1) 경로가 .nii.gz 파일인지 디렉토리인지 확인
        if path_or_dir.endswith(".nii.gz"):
            # 누락 디렉토리의 파일
            image_src = path_or_dir
        else:
            # 기본 디렉토리 구조
            sub_dir = os.path.join(path_or_dir, "0002")
            
            if not os.path.isdir(sub_dir):
                log_messages.append(f"0002 folder not found for patient: {pid}")
                continue
            
            # 이미지 파일 찾기
            image_files = [f for f in os.listdir(sub_dir)
                           if f.endswith(".nii.gz") and os.path.isfile(os.path.join(sub_dir, f))]
            
            if not image_files:
                log_messages.append(f"Image file not found for patient: {pid}")
                continue
                
            # 자연수 정렬 후 첫 번째 파일 선택
            image_files = natsorted(image_files)
            image_src = os.path.join(sub_dir, image_files[0])

        # 2) 이미지 파일이 존재하는지 확인
        if not image_src or not os.path.exists(image_src):
            log_messages.append(f"Image file not accessible for patient: {pid}")
            continue

        # 3) nnUNet 명명 규칙에 따른 파일 복사
        count += 1
        global_success_count += 1
        
        # nnUNet 표준 파일명: {CaseID}_{SequenceID}_0000.nii.gz (이미지)
        new_image_name = f"{pid}_{count:04d}_0000.nii.gz"
        
        # 실제 이미지 파일 복사 수행
        shutil.copy(image_src, os.path.join(images_target, new_image_name))
    
    return count

# --- STEP 5: 실제 파일 복사 수행 ---
print("\n=== STEP 5: 파일 복사 수행 ===")
print("모든 환자 데이터를 테스트 세트로 복사 중...")
copy_patient_files_total(all_valid_patients, imagesVal_dir, start_count=0)

# --- STEP 6: 처리 결과 로그 생성 ---
print("\n=== STEP 6: 처리 결과 로그 생성 ===")

# 통계 계산
total_cases = len(all_patient_info) + len(patient_files_missing)  # 발견된 전체 환자 수
test_count = len(all_valid_patients)                              # 테스트 환자 수
with_gt_count = len(valid_patients_with_gt)                       # GT가 있는 환자 수
without_gt_count = len(valid_patients_without_gt)                 # GT가 없는 환자 수
fail_count = total_cases - global_success_count                   # 실패한 환자 수

# 로그 메시지 작성
log_messages.append("Organizing KMU Cardiac_AVC TOTAL dataset in nnUNet format (test ratio 1.0) is complete.")
log_messages.append("\n=== Processing Summary ===")
log_messages.append(f"Total cases discovered: {total_cases}\n")
log_messages.append(f"From base directory: {len(patient_info_base)}")
log_messages.append(f"From missing directory: {len(patient_files_missing)}\n")
log_messages.append(f"Valid cases with GT: {with_gt_count}")
log_messages.append(f"- Large FoV with GT: {len(large_fov_with_gt)}")
log_messages.append(f"- Small FoV with GT: {len(small_fov_with_gt)}\n")
log_messages.append(f"Valid cases without GT: {without_gt_count}")
log_messages.append(f"- Large FoV without GT: {len(large_fov_without_gt)}")
log_messages.append(f"- Small FoV without GT: {len(small_fov_without_gt)}\n")
log_messages.append(f"Test cases (all): {test_count}")
log_messages.append(f"Successfully processed: {global_success_count}")
log_messages.append(f"Failed cases: {fail_count}\n")
log_messages.append("Note: Only image files are copied to imagesVal directory (no labels).")

# 로그 파일 저장
results_file = os.path.join(dst_base, "cardiac_avc_total_nnUNet_results.txt")
with open(results_file, "w", encoding="utf-8") as f:
    for msg in log_messages:
        f.write(msg + "\n")

print(f"처리 완료! 결과 로그: {results_file}")
print(f"성공: {global_success_count}명, 실패: {fail_count}명")
print(f"총 테스트 환자: {test_count}명 (GT 있음: {with_gt_count}명, GT 없음: {without_gt_count}명)")
print("Dataset001_KMU_Cardiac_AVC_TOTAL 생성이 완료되었습니다. (imagesVal만 생성)")