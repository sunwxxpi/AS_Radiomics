import os
import shutil

def clean_directories_final(root_dir, dry_run=True):
    """
    지정된 디렉토리 구조(KUDH/KUDH/Anonymous/A/stor/results) 내에서
    'A/stor/results' 폴더가 없는 하위 폴더 ('A')를 삭제합니다.

    Args:
        root_dir (str): 최상위 디렉토리 경로.
        dry_run (bool): True로 설정하면 실제 삭제를 수행하지 않고 대상 폴더만 출력합니다.
                        False로 설정하면 실제 삭제를 수행합니다.
    """
    print(f"'{root_dir}' 디렉토리에서 작업을 시작합니다. Dry run: {dry_run}")

    if not os.path.isdir(root_dir):
        print(f"오류: 최상위 디렉토리 '{root_dir}'를 찾을 수 없습니다.")
        return

    # 최상위 디렉토리 (예: rm_no_results_target) 내의 항목 순회
    for item_level0 in os.listdir(root_dir):
        path_level1_kudh1 = os.path.join(root_dir, item_level0)
        # 첫 번째 KUDH 디렉토리인지 확인 (예: .../rm_no_results_target/KUDH0186)
        if os.path.isdir(path_level1_kudh1) and item_level0.startswith("KUDH"):
            print(f"  KUDH Level 1 디렉토리 확인 중: {path_level1_kudh1}")

            # 첫 번째 KUDH 디렉토리 내부 항목 순회
            for item_level1 in os.listdir(path_level1_kudh1):
                path_level2_kudh2 = os.path.join(path_level1_kudh1, item_level1)
                # 두 번째 KUDH 디렉토리인지 확인 (예: .../KUDH0186/KUDH0186)
                # 제공된 경로에서는 두 번째 디렉토리 이름도 KUDHxxxx 형태였습니다.
                if os.path.isdir(path_level2_kudh2) and item_level1.startswith("KUDH"):
                    print(f"    KUDH Level 2 디렉토리 확인 중: {path_level2_kudh2}")

                    # 두 번째 KUDH 디렉토리 내부 항목 순회 (Anonymous 디렉토리 찾기)
                    for item_level2_anony in os.listdir(path_level2_kudh2):
                        path_level3_anonymous = os.path.join(path_level2_kudh2, item_level2_anony)
                        if os.path.isdir(path_level3_anonymous) and item_level2_anony.startswith("Anonymous"):
                            print(f"      Anonymous 디렉토리 확인 중: {path_level3_anonymous}")

                            # Anonymous 디렉토리 내부의 'A' 디렉토리 순회
                            for item_level3_a_dir in os.listdir(path_level3_anonymous):
                                path_level4_a_dir = os.path.join(path_level3_anonymous, item_level3_a_dir)
                                if os.path.isdir(path_level4_a_dir):
                                    # 'A/stor/results' 디렉토리 존재 여부 확인
                                    stor_dir_path = os.path.join(path_level4_a_dir, "stor")
                                    results_path_in_stor = os.path.join(stor_dir_path, "results")

                                    should_keep_A = os.path.isdir(stor_dir_path) and \
                                                    os.path.isdir(results_path_in_stor)
                                    
                                    path_check_details = (
                                        f"(stor 존재: {os.path.isdir(stor_dir_path)}, "
                                        f"stor/results 존재: {os.path.isdir(results_path_in_stor) if os.path.isdir(stor_dir_path) else 'N/A (stor 없음)'})"
                                    )

                                    if not should_keep_A:
                                        print(f"        - '{path_level4_a_dir}' 내에 'stor/results' 디렉토리가 없습니다. {path_check_details}")
                                        if not dry_run:
                                            try:
                                                shutil.rmtree(path_level4_a_dir)
                                                print(f"          SUCCESS: '{path_level4_a_dir}' 삭제 완료.")
                                            except OSError as e:
                                                print(f"          ERROR: '{path_level4_a_dir}' 삭제 중 오류 발생: {e}")
                                        else:
                                            print(f"          INFO (Dry Run): '{path_level4_a_dir}' 삭제 대상입니다.")
                                    else:
                                        print(f"        - '{path_level4_a_dir}' 내에 'stor/results' 디렉토리가 존재하여 유지합니다. {path_check_details}")
    print("작업 완료.")

if __name__ == "__main__":
    # ！！！ 중요: 여기에 사용자님이 제공해주신 실제 최상위 경로를 정확히 입력하세요. ！！！
    # 예시로 제공해주신 경로를 기반으로 설정합니다.
    target_directory = "/home/psw/AS_Radiomics/rm_no_results_target"

    # --- 중요 ---
    # 실제 삭제를 원하시면 `is_dry_run=False`로 변경하세요.
    # 처음에는 `is_dry_run=True`로 실행하여 삭제 대상을 확인하는 것을 강력히 권장합니다.
    # ----------------
    is_dry_run = False # True: 사전 실행 (삭제 안 함), False: 실제 삭제 실행

    abs_target_path = os.path.abspath(target_directory)
    print(f"스크립트가 대상으로 하는 절대 경로는 '{abs_target_path}' 입니다.")
    if not os.path.exists(abs_target_path):
        print(f"경고: 대상 경로 '{abs_target_path}' 가 존재하지 않습니다. 'target_directory' 설정을 확인해주세요.")
        exit()
    if not os.path.isdir(abs_target_path):
        print(f"경고: 대상 경로 '{abs_target_path}' 는 디렉토리가 아닙니다. 'target_directory' 설정을 확인해주세요.")
        exit()

    if not is_dry_run:
        confirmation = input(
            f"경고: '{target_directory}' (절대 경로: {abs_target_path}) 내부의 조건에 맞는 디렉토리를 실제로 삭제합니다. "
            "계속하시겠습니까? (yes/no): "
        )
        if confirmation.lower() != 'yes':
            print("작업이 취소되었습니다.")
            exit()

    clean_directories_final(target_directory, dry_run=is_dry_run)