import os
import json
import glob
import nibabel as nib
from datetime import datetime
from tqdm import tqdm

def crop_nifti_by_voxel_coords(
    input_path: str,
    output_path: str,
    min_coords: tuple[int, int, int],
    max_coords: tuple[int, int, int],
):
    """
    NIfTI 파일(.nii.gz)을 지정된 복셀 좌표로 잘라내어 저장합니다.

    Args:
        input_path (str): 원본 NIfTI 파일 경로.
        output_path (str): 잘라낸 결과를 저장할 NIfTI 파일 경로.
        min_coords (tuple[int, int, int]): 자를 영역의 최소 복셀 좌표 (x_min, y_min, z_min).
        max_coords (tuple[int, int, int]): 자를 영역의 최대 복셀 좌표 (x_max, y_max, z_max).
    """
    try:
        # 1. NIfTI 파일 로드
        print(f"원본 파일 로드 중: {input_path}")
        nii_image = nib.load(input_path)

        # 2. 영상 데이터와 Affine 행렬 추출
        # get_fdata()는 데이터를 float64 타입의 numpy 배열로 가져옵니다.
        original_data = nii_image.get_fdata()
        original_affine = nii_image.affine
        original_header = nii_image.header

        print(f"원본 영상 shape: {original_data.shape}")
        print(f"자르기 전 Affine:\n{original_affine}")

        x_min, y_min, z_min = min_coords
        # NumPy 슬라이싱은 끝 인덱스를 포함하지 않으므로 +1을 해줍니다.
        x_max, y_max, z_max = (
            max_coords[0] + 1,
            max_coords[1] + 1,
            max_coords[2] + 1,
        )

        # 3. NumPy 배열 슬라이싱을 이용한 데이터 Crop
        cropped_data = original_data[x_min:x_max, y_min:y_max, z_min:z_max]
        
        print(f"잘라낸 영상 shape: {cropped_data.shape}")

        # 4. Affine 행렬 업데이트 (매우 중요)
        # Crop으로 인해 복셀 좌표계의 원점이 이동했으므로,
        # 새로운 원점(0,0,0)이 기존 좌표계의 (x_min, y_min, z_min) 위치를 가리키도록
        # Affine 행렬의 translation(이동) 부분을 업데이트합니다.
        new_affine = original_affine.copy()
        # 원본 Affine의 회전/스케일링 부분과 이동 시작점(x_min, y_min, z_min)을 곱하여
        # 이동해야 할 실제 거리(mm)를 계산하고, 이를 기존 origin에 더해줍니다.
        new_affine[:3, 3] = original_affine[:3, 3] + original_affine[:3, :3] @ [x_min, y_min, z_min]
        
        print(f"자른 후 업데이트된 Affine:\n{new_affine}")

        # 5. 새로운 NIfTI 이미지 생성
        # 잘라낸 데이터와 업데이트된 Affine으로 새로운 NIfTI 객체를 만듭니다.
        cropped_nii_image = nib.Nifti1Image(
            cropped_data, new_affine, header=original_header
        )

        # 6. 결과 파일 저장
        # 저장할 경로의 디렉토리가 없으면 생성합니다.
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        nib.save(cropped_nii_image, output_path)
        print(f"성공적으로 파일을 저장했습니다: {output_path}")

    except FileNotFoundError:
        print(f"[에러] 파일을 찾을 수 없습니다: {input_path}")
    except Exception as e:
        print(f"[에러] 처리 중 오류가 발생했습니다: {e}")

def parse_mask_pos(mask_pos_str):
    """
    maskPos 문자열을 파싱하여 최소/최대 좌표를 반환합니다.
    
    Args:
        mask_pos_str (str): "(161,169,27)-(282,274,41)" 형식의 문자열
        
    Returns:
        tuple: (min_coords, max_coords) 튜플
    """
    try:
        # "(161,169,27)-(282,274,41)" -> ["(161,169,27)", "(282,274,41)"]
        parts = mask_pos_str.split('-')
        
        # "(161,169,27)" -> "161,169,27" -> [161, 169, 27]
        min_coords_str = parts[0].strip('()')
        min_coords = tuple(map(int, min_coords_str.split(',')))
        
        # "(282,274,41)" -> "282,274,41" -> [282, 274, 41]
        max_coords_str = parts[1].strip('()')
        max_coords = tuple(map(int, max_coords_str.split(',')))
        
        return min_coords, max_coords
    except Exception as e:
        print(f"[에러] maskPos 파싱 실패: {mask_pos_str}, 에러: {e}")
        return None, None

def find_0002_series_directories(base_dir):
    """
    base_dir 하위에서 모든 0002 시리즈 디렉토리를 찾습니다.
    
    Args:
        base_dir (str): 검색할 기본 디렉토리 경로
        
    Returns:
        list: 0002 시리즈 디렉토리 경로들의 리스트
    """
    series_0002_dirs = []
    
    # **/0002 패턴으로 검색
    pattern = os.path.join(base_dir, "**/0002")
    found_dirs = glob.glob(pattern, recursive=True)
    
    for dir_path in found_dirs:
        if os.path.isdir(dir_path):
            series_0002_dirs.append(dir_path)
    
    return series_0002_dirs

def process_0002_series(series_dir, output_base_dir, failed_cases):
    """
    하나의 0002 시리즈 디렉토리를 처리합니다.
    
    Args:
        series_dir (str): 0002 시리즈 디렉토리 경로
        output_base_dir (str): 결과를 저장할 기본 디렉토리
        failed_cases (list): 실패한 케이스들을 기록할 리스트
    """
    print(f"\n=== 처리 중인 시리즈: {series_dir} ===")
    
    # results 디렉토리에서 JSON 파일 찾기
    results_dir = os.path.join(series_dir, "results")
    if not os.path.exists(results_dir):
        error_msg = f"results 디렉토리가 없음: {results_dir}"
        print(f"[경고] {error_msg}")
        failed_cases.append(f"{series_dir} - {error_msg}")
        return False
    
    # JSON 파일 찾기
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    if not json_files:
        error_msg = f"JSON 파일이 없음: {results_dir}"
        print(f"[경고] {error_msg}")
        failed_cases.append(f"{series_dir} - {error_msg}")
        return False
    
    json_file = json_files[0]  # 첫 번째 JSON 파일 사용
    print(f"JSON 파일 사용: {json_file}")
    
    # JSON에서 maskPos 추출
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # JSON 구조에서 maskPos 찾기
        mask_pos = None
        for study in json_data.get("study", []):
            for series in study.get("series", []):
                for lesion in series.get("lesion", []):
                    if "maskPos" in lesion:
                        mask_pos = lesion["maskPos"]
                        break
                if mask_pos:
                    break
            if mask_pos:
                break
        
        if not mask_pos:
            error_msg = f"maskPos를 찾을 수 없음: {json_file}"
            print(f"[경고] {error_msg}")
            failed_cases.append(f"{series_dir} - {error_msg}")
            return False
            
        print(f"maskPos 발견: {mask_pos}")
        
        # maskPos 파싱
        min_coords, max_coords = parse_mask_pos(mask_pos)
        if min_coords is None or max_coords is None:
            error_msg = f"maskPos 파싱 실패: {mask_pos}"
            print(f"[에러] {error_msg}")
            failed_cases.append(f"{series_dir} - {error_msg}")
            return False
            
    except Exception as e:
        error_msg = f"JSON 파일 읽기 실패: {json_file}, 에러: {e}"
        print(f"[에러] {error_msg}")
        failed_cases.append(f"{series_dir} - {error_msg}")
        return False
    
    # 시리즈 디렉토리와 results 디렉토리에서 .nii.gz 파일들 찾기
    series_nii_files = glob.glob(os.path.join(series_dir, "*.nii.gz"))
    results_nii_files = glob.glob(os.path.join(results_dir, "*.nii.gz"))
    
    all_nii_files = []
    # 시리즈 디렉토리의 파일들 (출력 위치: 시리즈 디렉토리)
    for nii_file in series_nii_files:
        all_nii_files.append({
            'file_path': nii_file,
            'output_subdir': '',  # 시리즈 디렉토리 바로 하위
            'source_type': 'series'
        })
    
    # results 디렉토리의 파일들 (출력 위치: results 디렉토리)
    for nii_file in results_nii_files:
        all_nii_files.append({
            'file_path': nii_file,
            'output_subdir': 'results',  # results 하위 디렉토리
            'source_type': 'results'
        })
    
    if not all_nii_files:
        error_msg = f".nii.gz 파일이 없음: {series_dir} 및 {results_dir}"
        print(f"[경고] {error_msg}")
        failed_cases.append(f"{series_dir} - {error_msg}")
        return False
    
    print(f"발견된 .nii.gz 파일 수: 시리즈({len(series_nii_files)}) + results({len(results_nii_files)}) = {len(all_nii_files)}")
    
    # 출력 디렉토리 생성
    relative_path = os.path.relpath(series_dir, "/home/psw/AS_Radiomics/data/datasets_raw/KMU/cardiac CT")
    output_dir = os.path.join(output_base_dir, relative_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # results 출력 디렉토리도 생성
    results_output_dir = os.path.join(output_dir, "results")
    os.makedirs(results_output_dir, exist_ok=True)
    
    # 모든 .nii.gz 파일 처리
    file_success_count = 0
    for file_info in tqdm(all_nii_files, desc=f"  파일 처리", leave=False):
        nii_file = file_info['file_path']
        output_subdir = file_info['output_subdir']
        source_type = file_info['source_type']
        
        filename = os.path.basename(nii_file)
        name_without_ext = filename.replace('.nii.gz', '')
        output_filename = f"{name_without_ext}.nii.gz"
        
        # 출력 경로 결정
        if output_subdir:
            output_path = os.path.join(output_dir, output_subdir, output_filename)
        else:
            output_path = os.path.join(output_dir, output_filename)
        
        try:
            # crop 실행
            crop_nifti_by_voxel_coords(
                input_path=nii_file,
                output_path=output_path,
                min_coords=min_coords,
                max_coords=max_coords
            )
            file_success_count += 1
            print(f"    성공: {source_type}/{filename}")
        except Exception as e:
            error_msg = f"파일 crop 실패: {source_type}/{filename}, 에러: {e}"
            print(f"  [에러] {error_msg}")
            failed_cases.append(f"{series_dir}/{source_type}/{filename} - {error_msg}")
    
    print(f"  처리 완료: {file_success_count}/{len(all_nii_files)} 파일 성공")
    return file_success_count > 0

def save_failed_cases_log(failed_cases, output_base_dir):
    """
    실패한 케이스들을 텍스트 파일에 저장합니다.
    
    Args:
        failed_cases (list): 실패한 케이스들의 리스트
        output_base_dir (str): 결과를 저장할 기본 디렉토리
    """
    if not failed_cases:
        print("실패한 케이스가 없습니다.")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"crop_failed_cases_{timestamp}.txt"
    log_path = os.path.join(output_base_dir, log_filename)
    
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Crop 처리 실패 케이스 로그\n")
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"총 실패 케이스 수: {len(failed_cases)}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, case in enumerate(failed_cases, 1):
                f.write(f"{i:3d}. {case}\n")
        
        print(f"실패 케이스 로그 저장됨: {log_path}")
    except Exception as e:
        print(f"[에러] 실패 케이스 로그 저장 실패: {e}")

def main():
    """
    메인 실행 함수
    """
    # 설정
    BASE_CARDIAC_CT_DIR = "/home/psw/AS_Radiomics/data/datasets_raw/KMU/cardiac CT"
    OUTPUT_BASE_DIR = "/home/psw/AS_Radiomics/data/datasets_raw/KMU/cardiac CT_av_roi_cropped"
    
    print(f"기본 디렉토리: {BASE_CARDIAC_CT_DIR}")
    print(f"출력 디렉토리: {OUTPUT_BASE_DIR}")
    
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    # 실패 케이스 기록용 리스트
    failed_cases = []
    
    # 0002 시리즈 디렉토리들 찾기
    print("0002 시리즈 디렉토리 검색 중...")
    series_0002_dirs = find_0002_series_directories(BASE_CARDIAC_CT_DIR)
    
    if not series_0002_dirs:
        print("0002 시리즈 디렉토리를 찾을 수 없습니다.")
        return
    
    print(f"발견된 0002 시리즈 수: {len(series_0002_dirs)}")
    for dir_path in series_0002_dirs:
        print(f"  - {dir_path}")
    
    # 각 시리즈 처리 (tqdm 적용)
    success_count = 0
    print(f"\n0002 시리즈 처리 시작...")
    
    for series_dir in tqdm(series_0002_dirs, desc="시리즈 처리 진행률"):
        try:
            if process_0002_series(series_dir, OUTPUT_BASE_DIR, failed_cases):
                success_count += 1
        except Exception as e:
            error_msg = f"시리즈 처리 중 예외 발생: {e}"
            print(f"[에러] {error_msg}")
            failed_cases.append(f"{series_dir} - {error_msg}")
    
    # 결과 요약
    print(f"\n=== 처리 완료 ===")
    print(f"총 시리즈 수: {len(series_0002_dirs)}")
    print(f"성공한 시리즈 수: {success_count}")
    print(f"실패한 시리즈 수: {len(series_0002_dirs) - success_count}")
    print(f"총 실패 케이스 수: {len(failed_cases)}")
    print(f"결과 저장 디렉토리: {OUTPUT_BASE_DIR}")
    
    # 실패 케이스 로그 저장
    if failed_cases:
        save_failed_cases_log(failed_cases, OUTPUT_BASE_DIR)

if __name__ == '__main__':
    main()