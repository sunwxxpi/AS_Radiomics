import os
import sys
import tempfile
import shutil
import SimpleITK as sitk
from tqdm import tqdm
from natsort import natsorted

def capture_stderr(func, *args, **kwargs):
    original_stderr_fd = sys.stderr.fileno()
    with tempfile.TemporaryFile(mode='w+b') as tmpfile:
        saved_stderr_fd = os.dup(original_stderr_fd)
        os.dup2(tmpfile.fileno(), original_stderr_fd)
        try:
            result = func(*args, **kwargs)
            sys.stderr.flush()
            os.fsync(original_stderr_fd)
            tmpfile.seek(0)
            captured = tmpfile.read().decode('utf-8')
        finally:
            os.dup2(saved_stderr_fd, original_stderr_fd)
            os.close(saved_stderr_fd)
    return result, captured

def log_error(log_file, patient_id, error_message):
    error_message = error_message.rstrip("\n")
    with open(log_file, "a") as f:
        f.write(f"Patient {patient_id}: {error_message}\n\n")

def convert_dicom_to_nifti(dicom_dir, output_nifti_path, patient_id, log_file):
    try:
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)
        if not dicom_files:
            print(f"Warning: No DICOM files found in {dicom_dir}. Skipping conversion.")
            return False

        reader.SetFileNames(dicom_files)
        image, warning_message = capture_stderr(reader.Execute)

        if warning_message:
            warning_message = warning_message.rstrip("\n")
            log_error(log_file, patient_id, warning_message)
            return False

        sitk.WriteImage(image, output_nifti_path)
        return True
    except Exception as e:
        log_error(log_file, patient_id, f"Error during DICOM conversion: {str(e)}")
        print(f"Error converting DICOM in {dicom_dir}: {e}")
        return False

def process_results_directory(root, patient_id, dst_results_dir, allowed_files, log_file):
    os.makedirs(dst_results_dir, exist_ok=True)
    any_error = False
    for idx, file in enumerate(allowed_files):
        src_path = os.path.join(root, file)
        if not os.path.exists(src_path):
            continue
        dst_filename = f"{patient_id}_{idx:03}.nii.gz"
        dst_path = os.path.join(dst_results_dir, dst_filename)
        try:
            image = sitk.ReadImage(src_path)
            array = sitk.GetArrayFromImage(image)
            array[array == 255] = 1
            new_image = sitk.GetImageFromArray(array)
            new_image.CopyInformation(image)
            sitk.WriteImage(new_image, dst_path)
        except Exception as e:
            log_error(log_file, patient_id, f"Error processing {file} in {root}: {e}")
            print(f"Error processing {file} in {root}: {e}")
            any_error = True
    return any_error

def initialize_log_file(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w") as f:
        f.write("")

def write_summary_to_log(log_file, total, success, failure, failed_ids):
    with open(log_file, "a") as f:
        f.write("\n=== Summary ===\n")
        f.write(f"Total cases: {total}\n")
        f.write(f"Success: {success}\n")
        f.write(f"Failure: {failure}\n\n")
        f.write("Failed patient IDs: " + ", ".join(sorted(failed_ids)))

def copy_chest_ct_files():
    src_base = "/home/psw/AS_Radiomics/data/datasets_raw/KMU_raw/chest CT"
    dst_base = "/home/psw/AS_Radiomics/data/datasets_raw/KMU/chest CT"
    log_file = "/home/psw/AS_Radiomics/data/datasets_raw/KMU/chest_ct_processing_log.txt"
    initialize_log_file(log_file)

    os.makedirs(dst_base, exist_ok=True)
    total_cases, success_count, failure_count = 0, 0, 0
    failed_patients = set()
    patient_list = natsorted(os.listdir(src_base))

    for patient_id in tqdm(patient_list, desc="Processing Chest CT Patients"):
        src_patient_path = os.path.join(src_base, patient_id)
        if not os.path.isdir(src_patient_path):
            continue
        dst_patient_dir = os.path.join(dst_base, patient_id)
        os.makedirs(dst_patient_dir, exist_ok=True)
        patient_warning_occurred = False

        results_dirs = [root for root, _, _ in os.walk(src_patient_path) if os.path.basename(root) == "results"]
        for results_dir in results_dirs:
            dst_results_dir = os.path.join(dst_patient_dir, "results")
            allowed_files = [f"lesionAnnot3D-{i:03}.nii.gz" for i in range(3)]
            if process_results_directory(results_dir, patient_id, dst_results_dir, allowed_files, log_file):
                patient_warning_occurred = True

        for root, dirs, files in os.walk(src_patient_path):
            if any(f.lower().endswith('.dcm') for f in files):
                total_cases += 1
                nifti_path = os.path.join(dst_patient_dir, f"{patient_id}.nii.gz")
                if not convert_dicom_to_nifti(root, nifti_path, patient_id, log_file):
                    failure_count += 1
                    patient_warning_occurred = True
                else:
                    success_count += 1
                dirs[:] = []

        if patient_warning_occurred:
            shutil.rmtree(dst_patient_dir)
            failed_patients.add(patient_id)

    write_summary_to_log(log_file, total_cases, success_count, failure_count, failed_patients)

if __name__ == "__main__":
    copy_chest_ct_files()