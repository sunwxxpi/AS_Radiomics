import os
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import SimpleITK as sitk
from tqdm import tqdm
from natsort import natsorted


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


def log_error(log_file: Path, patient_id: str, error_message: str):
    """지정된 로그 파일에 오류 메시지를 추가합니다."""
    error_message = error_message.strip()
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"Patient {patient_id}: {error_message}\n\n")


def convert_dicom_to_nifti(
    dicom_dir: Path,
    output_nifti_path: Path,
    patient_id: str,
    log_file: Path
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
            log_message = f"Patient {patient_id}: {warning_message.strip()}"
            tqdm.write(log_message)
            log_error(log_file, patient_id, warning_message)
            return False

        sitk.WriteImage(image, str(output_nifti_path))
        return True
    except Exception as e:
        error_msg = f"Error during DICOM conversion: {e}"
        log_message = f"Patient {patient_id}: {error_msg}"
        tqdm.write(log_message)
        log_error(log_file, patient_id, error_msg)
        return False


def get_patient_directories(source_dir: Path) -> list:
    """소스 디렉토리에서 KUDH로 시작하는 환자 디렉토리 목록을 재귀적으로 찾아 반환합니다."""
    patient_dirs = []
    for item in source_dir.rglob("KUDH*"):
        if item.is_dir():
            patient_dirs.append(item)
    return natsorted(patient_dirs)


def main():
    """1차_누락_cardiac_ct_1차_누락의 DICOM 파일들을 NIfTI로 변환합니다."""
    
    # 경로 설정
    source_base = Path("/home/psw/AS_Radiomics/data/datasets_raw/KMU_raw/cardiac CT/cardiac_ct_1차_누락")
    dest_base = Path("/home/psw/AS_Radiomics/data/datasets_raw/KMU/cardiac CT/cardiac_ct_1차_누락")
    log_file = Path("/home/psw/AS_Radiomics/data/datasets_raw/KMU/cardiac_ct_1차_누락_processing_log.txt")
    
    # 대상 디렉토리 생성
    dest_base.mkdir(parents=True, exist_ok=True)
    
    # 로그 파일 초기화
    if log_file.exists():
        log_file.unlink()
    log_file.touch()
    
    # 환자 디렉토리 목록 가져오기
    patient_dirs = get_patient_directories(source_base)
    
    if not patient_dirs:
        print(f"No patient directories found in {source_base}")
        return
    
    print(f"Found {len(patient_dirs)} patient directories to process")
    
    successful_patients = []
    failed_patients = []
    
    # 각 환자 디렉토리 처리
    progress_bar = tqdm(patient_dirs, desc="Converting DICOM to NIfTI")
    for patient_dir in progress_bar:
        patient_id = patient_dir.name
        progress_bar.set_description(f"Processing {patient_id}")
        
        # 환자 디렉토리의 상대 경로를 기반으로 출력 경로 결정
        relative_path = patient_dir.relative_to(source_base)
        
        # 환자 디렉토리 바로 위의 부모 디렉토리 경로 유지
        parent_path = relative_path.parent
        output_subdir = dest_base / parent_path
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        output_nifti_path = output_subdir / f"{patient_id}.nii.gz"
        
        # 이미 변환된 파일이 있으면 건너뛰기
        if output_nifti_path.exists():
            tqdm.write(f"Patient {patient_id}: Already converted, skipping")
            successful_patients.append(patient_id)
            continue
        
        # DICOM to NIfTI 변환
        if convert_dicom_to_nifti(patient_dir, output_nifti_path, patient_id, log_file):
            successful_patients.append(patient_id)
        else:
            failed_patients.append(patient_id)
            # 실패한 파일이 있으면 제거
            if output_nifti_path.exists():
                output_nifti_path.unlink()
    
    # 결과 요약
    with log_file.open("a", encoding="utf-8") as f:
        f.write("\n=== Summary ===\n")
        f.write(f"Total cases: {len(patient_dirs)}\n")
        f.write(f"Success: {len(successful_patients)}\n")
        f.write(f"Failure (not copied): {len(failed_patients)}\n\n")
        if failed_patients:
            sorted_failed = ", ".join(sorted(failed_patients))
            f.write(f"Failed patient IDs: {sorted_failed}\n")
    
    print("\n=== Processing Complete ===")
    print(f"Total patients: {len(patient_dirs)}")
    print(f"Successful conversions: {len(successful_patients)}")
    print(f"Failed conversions: {len(failed_patients)}")
    
    if failed_patients:
        print(f"\nFailed patients: {', '.join(sorted(failed_patients))}")
    
    print(f"\nLog file saved to: {log_file}")
    print(f"NIfTI files saved to: {dest_base}")


if __name__ == "__main__":
    main()