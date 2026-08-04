import os
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, generate_binary_structure

def perform_and_save_dilation(
    input_path: str,
    output_path: str,
    iterations: int
) -> None:
    """
    NIfTI 파일을 로드하여 지정된 횟수만큼 팽창을 수행하고 결과를 저장합니다.

    Args:
        input_path (str): 원본 마스크 파일 경로.
        output_path (str): 결과를 저장할 파일 경로.
        iterations (int): 팽창을 반복할 횟수.
    """
    try:
        # 1. 원본 NIfTI 파일 로드
        print(f"'{input_path}' 파일을 로드 중...")
        original_img = nib.load(input_path)
        
        # 2. 데이터를 NumPy 배열로 가져오기 (마스크 데이터는 보통 정수형이므로 uint8로 변환하여 처리)
        original_data = original_img.get_fdata().astype(np.uint8)
        
        print(f"팽창 연산 수행 중 (반복 횟수: {iterations})...")
        # 3. 3D 공간, 26-연결성을 위한 구조 요소(Structuring Element) 생성 (Rank=3: 3D, connectivity=3: 26-연결성 (대각선 포함))
        structure = generate_binary_structure(rank=3, connectivity=3)
        
        # 4. 형태학적 팽창 수행
        dilated_data = binary_dilation(
            input=original_data,
            structure=structure,
            iterations=iterations
        )
        
        # 5. 결과 데이터를 원본 데이터 타입으로 변환
        dilated_data = dilated_data.astype(original_data.dtype)
        
        print(f"결과를 '{output_path}' 파일로 저장 중...")
        # 6. 새로운 NIfTI 이미지 생성 (원본의 공간 정보(affine, header)를 그대로 사용하여 위치와 방향이 어긋나지 않도록 함)
        dilated_img = nib.Nifti1Image(
            dataobj=dilated_data,
            affine=original_img.affine,
            header=original_img.header
        )
        
        # 7. 파일로 저장
        nib.save(dilated_img, output_path)
        print(f"저장 완료: '{output_path}'")
        
    except FileNotFoundError:
        print(f"[오류] 파일을 찾을 수 없습니다: '{input_path}'")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")


def main():
    """메인 실행 함수"""
    input_filename = "KUDH0154_000.nii.gz"
    
    # 입력 파일 존재 여부 확인
    if not os.path.exists(input_filename):
        print(f"[오류] 현재 폴더에 '{input_filename}' 파일이 없습니다.")
        print("스크립트를 실행하기 전에 파일을 준비해주세요.")
        return

    # 출력 파일 이름 설정
    base_name = input_filename.replace(".nii.gz", "")
    output_filename_1iter = f"{base_name}_dilated_1iter.nii.gz"
    output_filename_2iter = f"{base_name}_dilated_2iter.nii.gz"
    
    # 팽창 1회 수행 및 저장
    perform_and_save_dilation(
        input_path=input_filename,
        output_path=output_filename_1iter,
        iterations=1
    )
    
    print("-" * 30)
    
    # 팽창 2회 수행 및 저장
    perform_and_save_dilation(
        input_path=input_filename,
        output_path=output_filename_2iter,
        iterations=2
    )
    
    print("\n모든 작업이 완료되었습니다.")


if __name__ == "__main__":
    main()