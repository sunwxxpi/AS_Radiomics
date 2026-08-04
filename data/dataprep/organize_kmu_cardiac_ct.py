import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import List, Set, Tuple, Dict, Any

import SimpleITK as sitk
from tqdm import tqdm
from natsort import natsorted


# --- Config 설정 ---

CARDIAC_CONFIG: Dict[str, Any] = {
    # 원본 데이터가 위치한 기본 디렉토리
    "SRC_BASE": Path("/home/psw/AS_Radiomics/data/datasets_raw/KMU_raw/cardiac CT"),

    # 처리된 데이터가 저장될 기본 디렉토리
    "DST_BASE": Path("/home/psw/AS_Radiomics/data/datasets_raw/KMU/cardiac CT"),

    # 상세 로그 파일 경로
    "LOG_FILE": Path("/home/psw/AS_Radiomics/data/datasets_raw/KMU/cardiac_ct_processing_log.txt"),

    # 환자 디렉토리를 식별하는 접두사 (예: "KUDH")
    "PATIENT_ID_PREFIX": "KUDH",

    # 처리할 대상 시리즈 번호 목록
    "TARGET_SERIES_NUMBERS": ["0002"],

    # 시리즈 번호별로 처리할 특정 어노테이션 파일들을 매핑
    "RESULTS_FILE_MAPPING": {
        "0001": [
            "lesionAnnot2D-000.nii.gz",
            "lesionAnnot2D-001.nii.gz"
        ],
        "0002": [
            "lesionAnnot3D-000.nii.gz",
            "lesionAnnot3D.json"
        ],
        "0003": [
        ],
    },
}


# --- 오류 캡처 및 로깅 함수들 ---

def capture_stderr(func, *args, **kwargs) -> Tuple[any, str]:
    """
    주어진 함수의 stderr 출력을 캡처합니다.
    ITK와 같은 라이브러리의 저수준 경고를 잡는 데 유용합니다.
    """
    original_stderr_fd = sys.stderr.fileno()
    with tempfile.TemporaryFile(mode='w+b') as tmpfile:
        saved_stderr_fd = os.dup(original_stderr_fd)
        os.dup2(tmpfile.fileno(), original_stderr_fd)
        try:
            result = func(*args, **kwargs)
            sys.stderr.flush()
            if hasattr(os, 'fsync'):
                os.fsync(original_stderr_fd)
            tmpfile.seek(0)
            captured = tmpfile.read().decode('utf-8', errors='ignore')
        finally:
            os.dup2(saved_stderr_fd, original_stderr_fd)
            os.close(saved_stderr_fd)
    return result, captured


def log_error(log_file: Path, patient_id: str, error_message: str, patient_dir_info: str = None):
    """지정된 로그 파일에 오류 메시지를 추가합니다."""
    error_message = error_message.strip()
    patient_display = f"{patient_id} ({patient_dir_info})" if patient_dir_info else patient_id
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"Patient {patient_display}: {error_message}\n\n")


# --- 핵심 처리 함수들 ---

def convert_dicom_to_nifti(
    dicom_dir: Path,
    output_nifti_path: Path,
    patient_id: str,
    log_file: Path,
    patient_dir_info: str = None
) -> bool:
    """DICOM 시리즈를 NIfTI 파일로 변환합니다."""
    try:
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(str(dicom_dir))
        if not dicom_files:
            return False

        reader.SetFileNames(dicom_files)
        image, warning_message = capture_stderr(reader.Execute)

        if warning_message:
            # 경고 메시지를 콘솔과 로그 모두에 출력
            patient_display = f"{patient_id} ({patient_dir_info})" if patient_dir_info else patient_id
            log_message = f"Patient {patient_display}: {warning_message.strip()}"
            tqdm.write(log_message)
            log_error(log_file, patient_id, warning_message, patient_dir_info)
            return False

        sitk.WriteImage(image, str(output_nifti_path))
        return True
    except Exception as e:
        error_msg = f"Error during DICOM conversion: {e}"
        patient_display = f"{patient_id} ({patient_dir_info})" if patient_dir_info else patient_id
        log_message = f"Patient {patient_display}: {error_msg}"
        tqdm.write(log_message)
        log_error(log_file, patient_id, error_msg, patient_dir_info)
        return False


def process_results_directory(
    results_dir: Path,
    dst_results_dir: Path,
    patient_id: str,
    allowed_files: List[str],
    log_file: Path,
    series_number: str,
    patient_dir_info: str = None
) -> bool:
    """
    'results' 디렉토리에서 NIfTI 파일들을 처리하고 복사합니다.
    
    설정에서 지정된 파일이 소스에서 누락된 경우, 경고를 로그에 기록하고 
    건너뜁니다. 기존 파일을 처리하는 중에 오류가 발생한 경우에만
    프로세스를 실패로 간주합니다.
    """
    dst_results_dir.mkdir(parents=True, exist_ok=True)
    any_processing_error = False  # 파일 처리 중 오류만 추적
    
    # 파일 타입별 인덱스 관리
    nifti_idx = 0
    json_idx = 0

    for filename in allowed_files:
        src_path = results_dir / filename

        # 1. 소스 파일이 누락된 경우: 로그 기록 후 실패로 처리
        if not src_path.exists():
            missing_file_msg = f"Annotation file not found in series {series_number}, skipping: {filename}"
            patient_display = f"{patient_id} ({patient_dir_info})" if patient_dir_info else patient_id
            log_message = f"Patient {patient_display}: {missing_file_msg}"
            tqdm.write(log_message)
            log_error(log_file, patient_id, missing_file_msg, patient_dir_info)
            any_processing_error = True
            continue  # 다음 파일로 건너뜀

        # 2. 파일이 존재하는 경우: 파일 형식에 따라 처리
        try:
            if filename.endswith('.nii.gz'):
                # NIfTI 파일은 기존 처리 (255 -> 1 변환)
                dst_filename = f"{patient_id}_{nifti_idx:03d}.nii.gz"
                dst_path = dst_results_dir / dst_filename
                
                image = sitk.ReadImage(str(src_path))
                array = sitk.GetArrayFromImage(image)
                array[array == 255] = 1  # 255 값을 1로 변환

                new_image = sitk.GetImageFromArray(array)
                new_image.CopyInformation(image)

                sitk.WriteImage(new_image, str(dst_path))
                nifti_idx += 1
            else:
                # 기타 파일은 단순 복사 (파일 타입에 따라 별도 인덱스 사용)
                file_extension = ''.join(src_path.suffixes)
                dst_filename = f"{patient_id}_{json_idx:03d}{file_extension}"
                dst_path = dst_results_dir / dst_filename
                shutil.copy2(src_path, dst_path)
                json_idx += 1
                
        except Exception as e:
            # 기존 파일을 처리하는 중 오류는 실패로 간주
            error_msg = (
                f"Error processing existing file {filename} "
                f"in series {series_number} {results_dir}: {e}"
            )
            patient_display = f"{patient_id} ({patient_dir_info})" if patient_dir_info else patient_id
            log_message = f"Patient {patient_display}: {error_msg}"
            tqdm.write(log_message)
            log_error(log_file, patient_id, error_msg, patient_dir_info)
            any_processing_error = True

    # 처리 오류가 발생한 경우에만 False 반환
    return not any_processing_error


def get_patient_ids(base_path: Path, prefix: str) -> List[str]:
    """주어진 기본 경로에서 환자 ID 목록을 추출합니다."""
    patient_ids = set()
    for path_object in base_path.rglob(f"{prefix}*"):
        if path_object.is_dir():
            patient_ids.add(path_object.name)
    return natsorted(list(patient_ids))


def write_summary_to_log(
    log_file: Path,
    total: int,
    success: int,
    failure: int,
    failed_ids: Set[str]
):
    """처리 결과 요약을 로그 파일에 작성합니다."""
    with log_file.open("a", encoding="utf-8") as f:
        f.write("\n=== Summary ===\n")
        f.write(f"Total cases: {total}\n")
        f.write(f"Success: {success}\n")
        f.write(f"Failure (not copied): {failure}\n\n")
        if failed_ids:
            sorted_failed = ", ".join(sorted(list(failed_ids)))
            f.write(f"Failed patient IDs: {sorted_failed}\n")


def clean_relative_path(relative_path: Path, numbered_dir: str) -> Path:
    """
    불필요한 디렉토리를 제거하고 시리즈 이름을 간소화하여 상대 경로를 정리합니다.
    - Anonymous_ 디렉토리 제거
    - 0001_xxx를 0001로, 0002_xxx를 0002로 대체
    - 경로에서 'stor' 디렉토리 제거
    """
    parts = list(relative_path.parts)
    cleaned_parts = []
    
    for part in parts:
        # Anonymous_ 디렉토리 건너뜀
        if part.startswith("Anonymous_"):
            continue
            
        # 시리즈 디렉토리를 간소화된 이름으로 대체
        if part.startswith("0001_"):
            cleaned_parts.append("0001")
        elif part.startswith("0002_"):
            cleaned_parts.append("0002")
        elif part.startswith("0003_"):
            cleaned_parts.append("0003")
        # 'stor' 디렉토리 건너뜀
        elif part == "stor":
            continue
        else:
            cleaned_parts.append(part)
    
    return Path(*cleaned_parts) if cleaned_parts else Path(".")


def find_relevant_dirs(src_base: Path, patient_id: str) -> List[Tuple[Path, Path]]:
    """특정 환자와 관련된 모든 디렉토리와 상대 경로를 찾습니다."""
    relevant_dirs = []
    for p in src_base.rglob("*"):
        if p.is_dir() and patient_id in p.parts:
            # src_base로부터 상대 경로 계산
            relative_path = p.relative_to(src_base)
            relevant_dirs.append((p, relative_path))
    return relevant_dirs


def process_patient(patient_id: str, config: Dict[str, Any]) -> bool:
    """설정에 따라 단일 환자의 모든 파일을 처리합니다."""
    src_base: Path = config["SRC_BASE"]
    dst_base: Path = config["DST_BASE"]
    log_file: Path = config["LOG_FILE"]

    has_dicom_to_process = False
    patient_failed = False  # 환자 처리 실패 여부 추적
    processed_dirs = set()  # 중복 방지를 위해 처리된 디렉토리 추적
    failed_series = set()  # 정리를 위한 실패한 시리즈 추적
    
    # 발견된 시리즈 추적
    found_series = set()
    # 환자의 디렉토리 경로 정보 수집
    patient_dir_info = None

    for root, relative_path in find_relevant_dirs(src_base, patient_id):
        # 환자 디렉토리 정보 수집 (처음 발견시에만)
        if patient_dir_info is None:
            # 환자 ID가 포함된 부모 디렉토리들을 찾아서 경로 구성
            patient_path_parts = []
            for part in relative_path.parts:
                if patient_id in part:
                    break
                patient_path_parts.append(part)
            patient_dir_info = "/".join(patient_path_parts) if patient_path_parts else "root"

        # 000으로 시작하는 시리즈 디렉토리 부분 찾기
        numbered_dir_part = next(
            (part for part in root.parts if part.startswith("000")), None
        )
        if not numbered_dir_part:
            continue

        # 시리즈 번호 추출 (예: "0001_18000101_000000"에서 "0001")
        numbered_dir = numbered_dir_part.split("_")[0]
        if numbered_dir not in config["TARGET_SERIES_NUMBERS"]:
            continue

        # 이 시리즈를 발견됨으로 표시
        found_series.add(numbered_dir)

        # 불필요한 디렉토리를 제거하여 상대 경로 정리
        cleaned_relative_path = clean_relative_path(relative_path, numbered_dir)
        dest_dir = dst_base / cleaned_relative_path
        
        series_failed = False
        
        # 1. DICOM 디렉토리 처리
        if any(f.suffix.lower() == '.dcm' for f in root.glob("*")):
            has_dicom_to_process = True
            dest_dir.mkdir(parents=True, exist_ok=True)
            nifti_path = dest_dir / f"{patient_id}.nii.gz"
            
            if not convert_dicom_to_nifti(root, nifti_path, patient_id, log_file, patient_dir_info):
                series_failed = True
                patient_failed = True  # 환자 전체를 실패로 표시
                error_msg = f"DICOM conversion failed for series {numbered_dir}"
                patient_display = f"{patient_id} ({patient_dir_info})" if patient_dir_info else patient_id
                log_message = f"Patient {patient_display}: {error_msg}"
                tqdm.write(log_message)
                log_error(log_file, patient_id, error_msg, patient_dir_info)
                
                # 실패한 DICOM 파일 제거
                if nifti_path.exists():
                    nifti_path.unlink()
            
            processed_dirs.add(dest_dir)

        # 2. "results" 디렉토리 처리
        if root.name == "results":
            allowed_files = config["RESULTS_FILE_MAPPING"].get(numbered_dir, [])
            if not allowed_files:
                continue

            results_dest_dir = dest_dir
            results_dest_dir.mkdir(parents=True, exist_ok=True)
            
            if not process_results_directory(
                root, results_dest_dir, patient_id, allowed_files, log_file, numbered_dir, patient_dir_info
            ):
                series_failed = True
                patient_failed = True  # 환자 전체를 실패로 표시
                error_msg = f"Results processing failed for series {numbered_dir}"
                patient_display = f"{patient_id} ({patient_dir_info})" if patient_dir_info else patient_id
                log_message = f"Patient {patient_display}: {error_msg}"
                tqdm.write(log_message)
                log_error(log_file, patient_id, error_msg, patient_dir_info)
            
            processed_dirs.add(results_dest_dir)
        
        # 실패한 시리즈 추적 및 정리
        if series_failed:
            failed_series.add(numbered_dir)
            
            # 전체 실패한 시리즈 디렉토리 제거
            if dest_dir.exists():
                shutil.rmtree(dest_dir)

    # 누락된 시리즈 확인 및 경고 로그 (실패로 간주)
    missing_series = set(config["TARGET_SERIES_NUMBERS"]) - found_series
    for missing_series_num in missing_series:
        missing_series_msg = f"Series not found, skipping: {missing_series_num}"
        patient_display = f"{patient_id} ({patient_dir_info})" if patient_dir_info else patient_id
        log_message = f"Patient {patient_display}: {missing_series_msg}"
        tqdm.write(log_message)
        log_error(log_file, patient_id, missing_series_msg, patient_dir_info)
        patient_failed = True  # 누락된 시리즈도 환자 실패로 간주

    if not has_dicom_to_process:
        # DICOM이 처리되지 않은 경우 이 환자를 위해 생성된 디렉토리 제거
        for processed_dir in processed_dirs:
            if processed_dir.exists():
                # 환자별 디렉토리 찾기 (patient_id 포함)
                patient_root = None
                for parent in [processed_dir] + list(processed_dir.parents):
                    if parent.name == patient_id and parent.parent != dst_base:
                        patient_root = parent
                        break
                if patient_root and patient_root.exists():
                    shutil.rmtree(patient_root)
                    break
        # patient_failed 상태에 따라 반환 (시리즈가 누락된 경우 실패로 처리)
        return not patient_failed

    # 환자 처리 실패 여부 반환
    return not patient_failed


def main(config: Dict[str, Any]):
    """설정에 의해 제어되는 스크립트의 메인 실행 함수입니다."""
    # --- 설정에서 설정값 로드 ---
    src_base: Path = config["SRC_BASE"]
    dst_base: Path = config["DST_BASE"]
    log_file: Path = config["LOG_FILE"]
    patient_id_prefix: str = config["PATIENT_ID_PREFIX"]

    # --- 초기화 ---
    dst_base.mkdir(parents=True, exist_ok=True)
    if log_file.exists():
        log_file.unlink()
    log_file.touch()

    patient_ids = get_patient_ids(src_base, prefix=patient_id_prefix)
    successful_patients = set()
    failed_patients = set()

    # --- 환자별 처리 루프 ---
    if not patient_ids:
        print(
            f"No patients found with prefix '{patient_id_prefix}' "
            f"in '{src_base}'."
        )
        return

    progress_bar = tqdm(patient_ids, desc="Processing Cardiac CT Patients")
    for patient_id in progress_bar:
        is_success = process_patient(patient_id, config)
        # 처리할 DICOM이 없어서 건너뛴 경우에도 성공으로 간주
        if is_success:
            successful_patients.add(patient_id)
        else:
            failed_patients.add(patient_id)

    # --- 최종 요약 ---
    # 요약은 변환이 시도된 환자들만 반영해야 함
    attempted_patients = successful_patients.union(failed_patients)

    write_summary_to_log(
        log_file,
        total=len(attempted_patients),
        success=len(successful_patients),
        failure=len(failed_patients),
        failed_ids=failed_patients,
    )

    print("\nProcessing finished.")
    print(
        f"Success: {len(successful_patients)}, "
        f"Failure: {len(failed_patients)}"
    )


if __name__ == "__main__":
    main(CARDIAC_CONFIG)