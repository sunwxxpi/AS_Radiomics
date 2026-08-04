import pandas as pd

# CSV 파일 읽기 (파일 경로 및 인코딩 옵션을 파일에 맞게 수정)
df = pd.read_csv('/home/psw/AS_Radiomics/data/AS_CRF.csv')

# 필요한 날짜 컬럼들을 datetime 형식으로 변환 (오류 발생 시 결측치로 처리)
date_cols = ['Cardiac CT검사일자', 'Chest CT검사일자'] + [f'Chest PA date_{i}' for i in range(1, 7)]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# 검증을 위해 기존 CT 검사일자 컬럼을 복사해서 새로운 열로 추가
df['Cardiac CT검사일'] = df['Cardiac CT검사일자']
df['Chest CT검사일'] = df['Chest CT검사일자']

# 각 행별로 Chest PA 날짜 후보들 중, 알고리즘에 따라 Closest Chest PA date와 해당 열 이름 결정
def compute_closest(row):
    # CT 검사일자 읽기
    cardiac_date = row['Cardiac CT검사일자']
    chest_date = row['Chest CT검사일자']
    
    # Chest PA date_1 ~ date_6에서 값과 해당 열 이름을 함께 수집
    pa_candidates = []
    for i in range(1, 7):
        col_name = f'Chest PA date_{i}'
        if pd.notnull(row[col_name]):
            pa_candidates.append((row[col_name], col_name))
    if not pa_candidates:
        return pd.Series({
            'Closest Chest PA date': pd.NaT,
            'Chest PA - Cardiac CT': pd.NaT,
            'Chest PA - Chest CT': pd.NaT,
            'Closest Chest PA column': None
        })
    
    # 두 CT 날짜가 모두 존재하는 경우
    if pd.notnull(cardiac_date) and pd.notnull(chest_date):
        lower = min(cardiac_date, chest_date)
        upper = max(cardiac_date, chest_date)
        # 두 CT 사이에 위치한 Chest PA 날짜 후보들
        candidates_in_between = [(pa, col) for (pa, col) in pa_candidates if lower <= pa <= upper]
        if candidates_in_between:
            # CT 날짜의 중간값(midpoint)에 가장 가까운 후보 선택
            mid_point = lower + (upper - lower) / 2
            best_candidate = min(candidates_in_between, key=lambda x: abs(x[0] - mid_point))
        else:
            # 두 CT 사이에 없으면, 각 후보에 대해 더 가까운 CT와의 차이가 최소인 후보 선택
            best_candidate = min(pa_candidates, key=lambda x: min(abs(x[0] - cardiac_date), abs(x[0] - chest_date)))
    # 한쪽 CT만 존재하는 경우
    elif pd.notnull(cardiac_date):
        best_candidate = min(pa_candidates, key=lambda x: abs(x[0] - cardiac_date))
    elif pd.notnull(chest_date):
        best_candidate = min(pa_candidates, key=lambda x: abs(x[0] - chest_date))
    else:
        best_candidate = (pd.NaT, None)
    
    best_date, best_col = best_candidate
    diff_cardiac = abs(best_date - cardiac_date) if pd.notnull(cardiac_date) and pd.notnull(best_date) else pd.NaT
    diff_chest = abs(best_date - chest_date) if pd.notnull(chest_date) and pd.notnull(best_date) else pd.NaT
    
    return pd.Series({
        'Closest Chest PA date': best_date,
        'Chest PA - Cardiac CT': diff_cardiac,
        'Chest PA - Chest CT': diff_chest,
        'Closest Chest PA column': best_col
    })

# 새로 추가할 열: 'Cardiac CT & CXR', 'Chest CT & CXR', 'CT & CXR'
def check_within_365(row):
    # Cardiac CT & CXR: Cardiac CT검사일자와 Closest Chest PA date 비교
    if pd.notnull(row['Cardiac CT검사일자']) and pd.notnull(row['Closest Chest PA date']):
        if pd.notnull(row['Chest PA - Cardiac CT']) and row['Chest PA - Cardiac CT'] <= pd.Timedelta(days=365):
            cardiac_check = 'o'
        else:
            cardiac_check = 'x'
    else:
        cardiac_check = 'Nan'
    
    # Chest CT & CXR: Chest CT검사일자와 Closest Chest PA date 비교
    if pd.notnull(row['Chest CT검사일자']) and pd.notnull(row['Closest Chest PA date']):
        if pd.notnull(row['Chest PA - Chest CT']) and row['Chest PA - Chest CT'] <= pd.Timedelta(days=365):
            chest_check = 'o'
        else:
            chest_check = 'x'
    else:
        chest_check = 'Nan'
    
    # CT & CXR: 세 날짜(Cardiac CT, Chest CT, Closest Chest PA date)가 모두 존재할 때,
    # 세 날짜 중 최솟값과 최댓값의 차이가 365일 이하이면 'o', 아니면 'x'
    if pd.notnull(row['Cardiac CT검사일자']) and pd.notnull(row['Chest CT검사일자']) and pd.notnull(row['Closest Chest PA date']):
        lower_bound = min(row['Cardiac CT검사일자'], row['Chest CT검사일자'], row['Closest Chest PA date'])
        upper_bound = max(row['Cardiac CT검사일자'], row['Chest CT검사일자'], row['Closest Chest PA date'])
        if upper_bound - lower_bound <= pd.Timedelta(days=365):
            ct_cxr_check = 'o'
        else:
            ct_cxr_check = 'x'
    else:
        ct_cxr_check = 'Nan'
    
    return pd.Series({
        'Cardiac CT & CXR': cardiac_check,
        'Chest CT & CXR': chest_check,
        'CT & CXR': ct_cxr_check
    })

# 각 행에 함수 적용하여 새로운 열 생성
result = df.apply(compute_closest, axis=1)
df = pd.concat([df, result], axis=1)

additional_result = df.apply(check_within_365, axis=1)
df = pd.concat([df, additional_result], axis=1)

# 최종 CSV 파일로 저장하기 전에, 'Closest Chest PA column' 열을 'Closest Chest PA date' 바로 오른쪽으로 재정렬
cols = list(df.columns)
if 'Closest Chest PA date' in cols and 'Closest Chest PA column' in cols:
    idx = cols.index('Closest Chest PA date')
    cols.remove('Closest Chest PA column')
    cols.insert(idx+1, 'Closest Chest PA column')
    df = df[cols]

# 최종 CSV 파일로 저장
df.to_csv('AS_CRF_summary.csv', index=False, encoding='utf-8-sig')

# 총 행 수 및 각 조건에 따른 건수와 해당 행의 '1차년도연구번호', 'Closest Chest PA date' 및 해당 열 정보 추출
total_rows = df.shape[0]

# Cardiac CT & CXR만 'o'인 경우: Cardiac CT & CXR이 'o'이고 동시에 Chest CT & CXR은 'o'가 아닌 경우
cardiac_only_df = df[(df['Cardiac CT & CXR'] == 'o') & (df['Chest CT & CXR'] != 'o')]
cardiac_only = cardiac_only_df.shape[0]

# Chest CT & CXR만 'o'인 경우: Chest CT & CXR이 'o'이고 동시에 Cardiac CT & CXR은 'o'가 아닌 경우
chest_only_df = df[(df['Chest CT & CXR'] == 'o') & (df['Cardiac CT & CXR'] != 'o')]
chest_only = chest_only_df.shape[0]

# CT & CXR 'o'인 경우: 'CT & CXR' 컬럼이 'o'인 행
ct_cxr_df = df[df['CT & CXR'] == 'o']
ct_cxr_count = ct_cxr_df.shape[0]

# txt 파일에 결과 기록 (한글 인코딩 utf-8-sig 사용)
with open("AS_CRF_summary.txt", "w", encoding="utf-8-sig") as f:
    f.write("===== 전체 결과 =====\n")
    f.write("총 행 수: {}\n\n".format(total_rows))
    
    f.write("===== Cardiac CT & CXR만 o인 건 =====\n")
    f.write("건 수: {}\n".format(cardiac_only))
    f.write("해당 '1차년도연구번호', 'Closest Chest PA date' 및 해당 열 목록:\n")
    for idx, row in cardiac_only_df.iterrows():
        date_str = row['Closest Chest PA date'].strftime('%Y-%m-%d') if pd.notnull(row['Closest Chest PA date']) else 'NaT'
        col_str = row['Closest Chest PA column'] if row['Closest Chest PA column'] is not None else 'NaT'
        f.write(f"{row['1차년도연구번호']} - {date_str} ({col_str})\n")
    
    f.write("\n===== Chest CT & CXR만 o인 건 =====\n")
    f.write("건 수: {}\n".format(chest_only))
    f.write("해당 '1차년도연구번호', 'Closest Chest PA date' 및 해당 열 목록:\n")
    for idx, row in chest_only_df.iterrows():
        date_str = row['Closest Chest PA date'].strftime('%Y-%m-%d') if pd.notnull(row['Closest Chest PA date']) else 'NaT'
        col_str = row['Closest Chest PA column'] if row['Closest Chest PA column'] is not None else 'NaT'
        f.write(f"{row['1차년도연구번호']} - {date_str} ({col_str})\n")
    
    f.write("\n===== CT & CXR이 o인 건 =====\n")
    f.write("건 수: {}\n".format(ct_cxr_count))
    f.write("해당 '1차년도연구번호', 'Closest Chest PA date' 및 해당 열 목록:\n")
    for idx, row in ct_cxr_df.iterrows():
        date_str = row['Closest Chest PA date'].strftime('%Y-%m-%d') if pd.notnull(row['Closest Chest PA date']) else 'NaT'
        col_str = row['Closest Chest PA column'] if row['Closest Chest PA column'] is not None else 'NaT'
        f.write(f"{row['1차년도연구번호']} - {date_str} ({col_str})\n")

print("총 행 수:", total_rows)
print("Cardiac CT & CXR만 o인 건 수:", cardiac_only)
print("Chest CT & CXR만 o인 건 수:", chest_only)
print("CT & CXR이 o인 건 수:", ct_cxr_count)
print("요약 정보가 summary.txt 파일에 저장되었습니다.")