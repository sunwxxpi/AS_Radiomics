import os
import numpy as np
import nibabel as nib

def extract_minimal_bounding_box(original_img_path, annotation_img_path, output_path=None):
    """
    원본 이미지에서 주석 파일이 표시하는 영역을 포함하는 최소한의 3D 경계 상자를 추출합니다.
    
    Args:
        original_img_path (str): 원본 이미지 경로
        annotation_img_path (str): 병변 주석 이미지 경로
        output_path (str, optional): 출력 파일 경로. 기본값은 None으로, 설정되지 않으면 자동 생성합니다.
    
    Returns:
        tuple: (추출된 이미지 데이터, 추출된 이미지 객체, 경계 상자 좌표)
    """
    # 원본 이미지와 주석 이미지 로드
    orig_img = nib.load(original_img_path)
    annot_img = nib.load(annotation_img_path)
    
    # 이미지 데이터 추출
    orig_data = orig_img.get_fdata()
    annot_data = annot_img.get_fdata()
    
    # 병변 영역(0이 아닌 값)의 좌표 찾기
    non_zero_indices = np.where(annot_data > 0)
    
    # 경계 상자 좌표 계산
    if len(non_zero_indices[0]) == 0:
        raise ValueError("병변 주석 이미지에 표시된 영역이 없습니다.")
    
    min_x, max_x = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
    min_y, max_y = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
    min_z, max_z = np.min(non_zero_indices[2]), np.max(non_zero_indices[2])
    
    # Z 좌표가 올바른지 확인 (최소값이 최대값보다 작아야 함)
    if min_z > max_z:
        min_z, max_z = max_z, min_z  # 순서가 뒤바뀌었다면 교환
    
    # 약간의 여백 추가 (선택 사항)
    padding = 5
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    min_z = max(0, min_z - padding)
    max_x = min(orig_data.shape[0] - 1, max_x + padding)
    max_y = min(orig_data.shape[1] - 1, max_y + padding)
    max_z = min(orig_data.shape[2] - 1, max_z + padding)
    
    # 각 차원의 크기가 최소 1 이상이 되도록 보장
    if max_x <= min_x:
        max_x = min_x + 1
    if max_y <= min_y:
        max_y = min_y + 1
    if max_z <= min_z:
        max_z = min_z + 1
    
    # 경계 상자 내의 데이터 추출
    cropped_data = orig_data[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]
    
    # 추출된 이미지 생성
    cropped_img = nib.Nifti1Image(cropped_data, orig_img.affine)
    
    # 출력 경로 설정
    if output_path is None:
        fname, ext = os.path.splitext(os.path.basename(original_img_path))
        if ext == '.gz':  # .nii.gz 파일 처리
            fname = os.path.splitext(fname)[0]
        output_path = f"{fname}_cropped.nii.gz"
    
    # 이미지 저장
    nib.save(cropped_img, output_path)
    
    print(f"원본 이미지 크기: {orig_data.shape}")
    print(f"추출된 영역 크기: {cropped_data.shape}")
    print(f"경계 상자 좌표: X[{min_x}:{max_x}], Y[{min_y}:{max_y}], Z[{min_z}:{max_z}]")
    print(f"추출된 이미지가 '{output_path}'에 저장되었습니다.")
    
    return cropped_data, cropped_img, ((min_x, max_x), (min_y, max_y), (min_z, max_z))

if __name__ == "__main__":
    orig_path = "./KUDH0004.nii.gz"
    annot_path = "./KUDH0004_000.nii.gz"
    
    extract_minimal_bounding_box(orig_path, annot_path)
