import os
import argparse
import numpy as np
import nibabel as nib
from datetime import datetime
from tqdm import tqdm

def crop_avc_images(image_dir, pred_dir, output_dir, crop_size=(160, 160, 32)):
    """
    AVC prediction mask를 기반으로 CT 영상을 crop하는 함수
    
    Args:
        image_dir: CT 영상 디렉토리 경로
        pred_dir: AVC prediction mask 디렉토리 경로  
        output_dir: crop된 결과를 저장할 디렉토리 경로
        crop_size: crop 크기 (W, H, D)
    """
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "imagesVal"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "predVal"), exist_ok=True)
    
    # 로그 파일 초기화
    log_file = os.path.join(output_dir, "avc_cropping_log.txt")
    
    # prediction 파일들 가져오기
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.nii.gz') and f != 'dataset.json'])
    
    print(f"=== AVC 기반 이미지 Cropping 시작 ===")
    print(f"Crop 크기: {crop_size}")
    print(f"총 처리할 케이스 수: {len(pred_files)}")
    print(f"출력 디렉토리: {output_dir}")
    
    # 통계 변수
    successful_crops = 0
    no_avc_cases = []
    avc_cases = []
    partial_crop_cases = []
    boundary_adjusted_cases = []
    error_cases = []
    
    # 로그 파일 헤더 작성
    with open(log_file, 'w') as f:
        f.write(f"AVC Cropping Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Crop Size: {crop_size}\n")
        f.write("="*80 + "\n\n")
    
    # tqdm 진행률 표시
    for pred_file in tqdm(pred_files, desc="처리 중", unit="케이스"):
        case_id = pred_file.replace('.nii.gz', '')
        img_file = case_id + '_0000.nii.gz'
        
        img_path = os.path.join(image_dir, img_file)
        pred_path = os.path.join(pred_dir, pred_file)
        
        # 파일 존재 확인
        if not os.path.exists(img_path):
            with open(log_file, 'a') as f:
                f.write(f"WARNING: 이미지 파일 없음 - {case_id}: {img_file}\n")
            continue
            
        try:
            # 이미지와 prediction 로드
            img_nii = nib.load(img_path)
            pred_nii = nib.load(pred_path)
            
            img_data = img_nii.get_fdata()
            pred_data = pred_nii.get_fdata()
            
            # AVC 마스크 분석
            avc_mask = (pred_data == 1)
            avc_volume = np.sum(avc_mask)
            
            if avc_volume == 0:
                no_avc_cases.append({
                    'case_id': case_id,
                    'image_shape': img_data.shape,
                    'crop_method': 'image_center'
                })
                # 원본 이미지 중심점 사용
                weighted_centroid = [img_data.shape[i] / 2 for i in range(3)]
                min_coords = max_coords = bbox_size = None  # AVC가 없으므로 바운딩 박스 없음
            else:
                # Weighted centroid 계산
                coords = np.where(avc_mask)
                weighted_centroid = [np.mean(coords[i]) for i in range(3)]
                
                # AVC 바운딩 박스 계산 (로깅용)
                min_coords = [np.min(coords[i]) for i in range(3)]
                max_coords = [np.max(coords[i]) for i in range(3)]
                bbox_size = [max_coords[i] - min_coords[i] + 1 for i in range(3)]
                
                # AVC 케이스 정보 저장
                avc_cases.append({
                    'case_id': case_id,
                    'avc_volume': avc_volume,
                    'bbox_size': bbox_size,
                    'weighted_centroid': weighted_centroid,
                    'image_shape': img_data.shape
                })
            
            # Crop 영역 계산
            half_crop = [crop_size[i] // 2 for i in range(3)]
            crop_start = [int(weighted_centroid[i] - half_crop[i]) for i in range(3)]
            crop_end = [crop_start[i] + crop_size[i] for i in range(3)]
            
            # 이미지 경계 체크 및 조정
            img_shape = img_data.shape
            original_crop_start = crop_start.copy()
            boundary_adjusted = False
            
            for i in range(3):
                if crop_start[i] < 0:
                    crop_start[i] = 0
                    crop_end[i] = crop_size[i]
                    boundary_adjusted = True
                elif crop_end[i] > img_shape[i]:
                    crop_end[i] = img_shape[i]
                    crop_start[i] = img_shape[i] - crop_size[i]
                    boundary_adjusted = True
                    
                # 여전히 범위를 벗어나는 경우 (이미지가 crop보다 작은 경우)
                if crop_start[i] < 0:
                    crop_start[i] = 0
                if crop_end[i] > img_shape[i]:
                    crop_end[i] = img_shape[i]
            
            if boundary_adjusted:
                boundary_adjusted_cases.append({
                    'case_id': case_id,
                    'original_crop_start': original_crop_start,
                    'adjusted_crop_start': crop_start,
                    'image_shape': img_data.shape
                })
            
            # Crop 수행
            cropped_img = img_data[crop_start[0]:crop_end[0], 
                                  crop_start[1]:crop_end[1], 
                                  crop_start[2]:crop_end[2]]
            cropped_mask = pred_data[crop_start[0]:crop_end[0], 
                                    crop_start[1]:crop_end[1], 
                                    crop_start[2]:crop_end[2]]
            
            # AVC가 있는 경우에만 완전 포함 여부 확인
            if avc_volume > 0:
                avc_fully_contained = (
                    min_coords[0] >= crop_start[0] and max_coords[0] < crop_end[0] and
                    min_coords[1] >= crop_start[1] and max_coords[1] < crop_end[1] and
                    min_coords[2] >= crop_start[2] and max_coords[2] < crop_end[2]
                )
                
                # Crop된 영역에서 AVC 복셀 수 확인
                cropped_avc_volume = np.sum(cropped_mask == 1)
                avc_loss_ratio = (avc_volume - cropped_avc_volume) / avc_volume * 100
                
                if not avc_fully_contained or avc_loss_ratio > 0.1:  # 0.1% 이상 손실
                    partial_crop_cases.append({
                        'case_id': case_id,
                        'original_avc_volume': avc_volume,
                        'cropped_avc_volume': cropped_avc_volume,
                        'loss_ratio': avc_loss_ratio,
                        'bbox_size': bbox_size,
                        'centroid': weighted_centroid,
                        'crop_region': f"[{crop_start[0]}:{crop_end[0]}, {crop_start[1]}:{crop_end[1]}, {crop_start[2]}:{crop_end[2]}]",
                        'avc_bbox': f"[{min_coords[0]}:{max_coords[0]}, {min_coords[1]}:{max_coords[1]}, {min_coords[2]}:{max_coords[2]}]"
                    })
            
            # 결과 저장
            # 새로운 NIfTI 이미지 생성 (원본 헤더 정보 유지)
            cropped_img_nii = nib.Nifti1Image(cropped_img, img_nii.affine, img_nii.header)
            cropped_mask_nii = nib.Nifti1Image(cropped_mask, pred_nii.affine, pred_nii.header)
            
            # 파일 저장 (원본 파일명 유지)
            img_output_path = os.path.join(output_dir, "imagesVal", img_file)
            mask_output_path = os.path.join(output_dir, "predVal", pred_file)
            
            nib.save(cropped_img_nii, img_output_path)
            nib.save(cropped_mask_nii, mask_output_path)
            
            successful_crops += 1
            
        except Exception as e:
            error_cases.append({
                'case_id': case_id,
                'error_message': str(e),
                'img_file': img_file
            })
    
    # 최종 로그 작성
    total_cases = len(pred_files)
    avc_detected_count = len(avc_cases)
    no_avc_count = len(no_avc_cases)
    
    print(f"\n=== 처리 완료 ===")
    print(f"총 처리된 케이스: {total_cases}")
    print(f"성공적으로 crop된 케이스: {successful_crops}")
    print(f"AVC 감지된 케이스: {avc_detected_count}")
    print(f"AVC 없는 케이스: {no_avc_count}")
    print(f"경계 조정된 케이스: {len(boundary_adjusted_cases)}")
    print(f"AVC 일부 누락 케이스: {len(partial_crop_cases)}")
    print(f"오류 케이스: {len(error_cases)}")
    
    # 상세 로그 저장
    with open(log_file, 'a') as f:
        f.write("=" * 80 + "\n")
        f.write("처리 결과 요약\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"📊 전체 통계\n")
        f.write(f"  • 총 처리 대상 케이스: {total_cases}개\n")
        f.write(f"  • 성공적으로 처리된 케이스: {successful_crops}개\n")
        f.write(f"  • 오류 케이스: {len(error_cases)}개\n\n")
        
        f.write(f"🎯 AVC 감지 현황\n")
        f.write(f"  • AVC 감지된 케이스: {avc_detected_count}개 ({avc_detected_count/total_cases*100:.1f}%)\n")
        f.write(f"  • AVC 없는 케이스: {no_avc_count}개 ({no_avc_count/total_cases*100:.1f}%)\n\n")
        
        f.write(f"⚙️ Crop 처리 현황\n")
        f.write(f"  • 이미지 경계 조정 케이스: {len(boundary_adjusted_cases)}개\n")
        f.write(f"  • AVC 일부 누락 케이스: {len(partial_crop_cases)}개\n\n")
        
        # AVC 통계 분석
        if avc_cases:
            volumes = [case['avc_volume'] for case in avc_cases]
            bbox_sizes = [case['bbox_size'] for case in avc_cases]
            
            f.write(f"📈 AVC 통계 분석\n")
            f.write(f"  • 평균 AVC 복셀 수: {np.mean(volumes):.1f}\n")
            f.write(f"  • AVC 복셀 수 범위: {np.min(volumes)} ~ {np.max(volumes)}\n")
            f.write(f"  • 평균 바운딩 박스 크기: [{np.mean([b[0] for b in bbox_sizes]):.1f}, {np.mean([b[1] for b in bbox_sizes]):.1f}, {np.mean([b[2] for b in bbox_sizes]):.1f}]\n")
            f.write(f"  • 최대 바운딩 박스 크기: [{np.max([b[0] for b in bbox_sizes])}, {np.max([b[1] for b in bbox_sizes])}, {np.max([b[2] for b in bbox_sizes])}]\n\n")
        
        # 각 카테고리별 상세 정보
        if no_avc_cases:
            f.write("=" * 50 + "\n")
            f.write("AVC 없는 케이스 (이미지 중심점 기반 Crop)\n")
            f.write("=" * 50 + "\n")
            for i, case_info in enumerate(no_avc_cases, 1):
                f.write(f"{i:2d}. {case_info['case_id']}\n")
                f.write(f"     원본 이미지 크기: {case_info['image_shape']}\n")
                f.write(f"     출력 파일: {case_info['case_id']}_0000.nii.gz / {case_info['case_id']}.nii.gz\n\n")
        
        if boundary_adjusted_cases:
            f.write("=" * 50 + "\n")
            f.write("이미지 경계 조정이 필요했던 케이스\n")
            f.write("=" * 50 + "\n")
            for i, case_info in enumerate(boundary_adjusted_cases, 1):
                f.write(f"{i:2d}. {case_info['case_id']}\n")
                f.write(f"     이미지 크기: {case_info['image_shape']}\n")
                f.write(f"     원본 crop 시작점: {case_info['original_crop_start']}\n")
                f.write(f"     조정된 crop 시작점: {case_info['adjusted_crop_start']}\n")
                f.write(f"     출력 파일: {case_info['case_id']}_0000.nii.gz / {case_info['case_id']}.nii.gz\n\n")
        
        if partial_crop_cases:
            f.write("=" * 50 + "\n")
            f.write("AVC 일부 누락된 케이스 (False Positive 가능성)\n")
            f.write("=" * 50 + "\n")
            for i, case_info in enumerate(partial_crop_cases, 1):
                f.write(f"{i:2d}. {case_info['case_id']}\n")
                f.write(f"     원본 AVC 복셀 수: {case_info['original_avc_volume']}\n")
                f.write(f"     Crop된 AVC 복셀 수: {case_info['cropped_avc_volume']}\n")
                f.write(f"     손실률: {case_info['loss_ratio']:.2f}%\n")
                f.write(f"     AVC 바운딩 박스 크기: {case_info['bbox_size']}\n")
                f.write(f"     Weighted centroid: {[f'{c:.1f}' for c in case_info['centroid']]}\n")
                f.write(f"     Crop 영역: {case_info['crop_region']}\n")
                f.write(f"     AVC 바운딩 박스: {case_info['avc_bbox']}\n")
                f.write(f"     출력 파일: {case_info['case_id']}_0000.nii.gz / {case_info['case_id']}.nii.gz\n\n")
        
        if error_cases:
            f.write("=" * 50 + "\n")
            f.write("오류가 발생한 케이스\n")
            f.write("=" * 50 + "\n")
            for i, case_info in enumerate(error_cases, 1):
                f.write(f"{i:2d}. {case_info['case_id']}\n")
                f.write(f"     이미지 파일: {case_info['img_file']}\n")
                f.write(f"     오류 내용: {case_info['error_message']}\n\n")
    
    print(f"상세 로그가 저장되었습니다: {log_file}")

def main():
    parser = argparse.ArgumentParser(description='AVC prediction 기반 CT 이미지 cropping')
    parser.add_argument('--image_dir', type=str, 
                       default='/home/psw/AS_Radiomics/data/datasets/Dataset001_KMU_Cardiac_AVC_TOTAL/imagesVal')
    parser.add_argument('--pred_dir', type=str,
                       default='/home/psw/AS_Radiomics/data/datasets/Dataset001_KMU_Cardiac_AVC_TOTAL/predVal')
    parser.add_argument('--output_dir', type=str,
                       default='/home/psw/AS_Radiomics/data/datasets/Dataset001_KMU_Cardiac_AVC_TOTAL_CROPPED')
    parser.add_argument('--crop_size', nargs=3, type=int, default=[160, 160, 32])
    
    args = parser.parse_args()
    
    crop_avc_images(
        image_dir=args.image_dir,
        pred_dir=args.pred_dir, 
        output_dir=args.output_dir,
        crop_size=tuple(args.crop_size)
    )

if __name__ == "__main__":
    main()