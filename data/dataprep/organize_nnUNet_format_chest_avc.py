import os
import shutil
from natsort import natsorted

# Define source and destination directories
src_base = "/home/psw/AS_Radiomics/data/datasets_raw/KMU/chest CT"
dst_base = "/home/psw/AS_Radiomics/data/datasets/Dataset002_KMU_Chest_AVC"

# Create nnUNet folder structure for images and labels
imagesTr_dir = os.path.join(dst_base, "imagesTr")
imagesVal_dir = os.path.join(dst_base, "imagesVal")
labelsTr_dir = os.path.join(dst_base, "labelsTr")
labelsVal_dir = os.path.join(dst_base, "labelsVal")
for d in (imagesTr_dir, imagesVal_dir, labelsTr_dir, labelsVal_dir):
    os.makedirs(d, exist_ok=True)

# Get list of all patient directories and sort them naturally
patient_dirs = [d for d in os.listdir(src_base) if os.path.isdir(os.path.join(src_base, d))]
patient_dirs = natsorted(patient_dirs)

# Filter patients with valid GT files (for splitting only) and log missing cases
log_messages = []
global_success_count = 0
valid_patients = []
for patient in patient_dirs:
    results_dir = os.path.join(src_base, patient, "results")
    if not os.path.isdir(results_dir):
        log_messages.append(f"'results' folder not found for patient: {patient}")
        continue
    gt_files = [f for f in os.listdir(results_dir) if f.endswith('_000.nii.gz')]
    if not gt_files:
        log_messages.append(f"GT file not found for patient: {patient}")
        continue
    valid_patients.append(patient)

# Split valid patients into training and validation (80:20 ratio)
split_index = int(len(valid_patients) * 0.8)
train_patients = valid_patients[:split_index]
test_patients  = valid_patients[split_index:]

def copy_patient_files(patient_list, images_target, labels_target):
    global global_success_count
    count = 0
    for patient in patient_list:
        patient_dir = os.path.join(src_base, patient)
        
        # 1) Image 파일 찾기
        image_files = [f for f in os.listdir(patient_dir) 
                       if f.endswith('.nii.gz') and os.path.isfile(os.path.join(patient_dir, f))]
        if not image_files:
            log_messages.append(f"Image file not found for patient: {patient}")
            continue
        image_files = natsorted(image_files)
        image_src = os.path.join(patient_dir, image_files[0])
        
        # 2) GT 파일 찾기
        results_dir = os.path.join(patient_dir, "results")
        gt_files = [f for f in os.listdir(results_dir) if f.endswith('_000.nii.gz')]
        if not gt_files:
            log_messages.append(f"GT file not found for patient: {patient}")
            continue
        gt_files = natsorted(gt_files)
        gt_src = os.path.join(results_dir, gt_files[0])
        
        # 3) 복사 및 이름 부여
        count += 1
        global_success_count += 1
        new_image_name = f"{patient}_{count:04d}_0000.nii.gz"
        new_label_name = f"{patient}_{count:04d}.nii.gz"
        shutil.copy(image_src, os.path.join(images_target, new_image_name))
        shutil.copy(gt_src,    os.path.join(labels_target, new_label_name))

# Copy files for training and validation datasets
copy_patient_files(train_patients, imagesTr_dir, labelsTr_dir)
copy_patient_files(test_patients,  imagesVal_dir, labelsVal_dir)

# Summary: total_cases = 전체 patient_dirs, fail_count = total – 성공 복사 수
total_cases = len(patient_dirs)
tr_count    = len(train_patients)
val_count   = len(test_patients)
fail_count  = total_cases - global_success_count

log_messages.append("Organizing KMU Chest_AVC dataset in nnUNet format is complete.")
log_messages.append("\n=== Summary ===")
log_messages.append(f"Total cases: {total_cases}")
log_messages.append(f"Tr cases: {tr_count}")
log_messages.append(f"Val cases: {val_count}")
log_messages.append(f"Failure: {fail_count}")

results_file = os.path.join(dst_base, "chest_avc_nnUNet_results.txt")
with open(results_file, "w") as f:
    for msg in log_messages:
        f.write(msg + "\n")

print(f"Results logged to {results_file}")