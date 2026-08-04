import pandas as pd
import numpy as np

# CSV 파일 읽기
df = pd.read_csv('/home/psw/AS_Radiomics/data/AS_CRF.csv')

# 날짜 컬럼을 datetime 형식으로 변환
date_cols = ['Cardiac CT검사일자', 'Chest CT검사일자', 'Echocardiography_date']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# 세 검사일자 간 간격이 모두 1년(365일) 이내인 케이스 필터링 함수
def is_within_365_days(row):
    dates = [row['Cardiac CT검사일자'], row['Chest CT검사일자'], row['Echocardiography_date']]
    # 결측값이 있는지 확인
    if any(pd.isnull(date) for date in dates):
        return False
    
    # 최소 날짜와 최대 날짜 간의 차이가 365일 이내인지 확인
    min_date = min(dates)
    max_date = max(dates)
    return (max_date - min_date).days <= 365

# 세 검사일자 간 간격이 모두 1년 이내인 케이스 필터링
df['Within_365_days'] = df.apply(is_within_365_days, axis=1)
filtered_df = df[df['Within_365_days']]

# AS 라벨 표준화 (대소문자 통일, 앞뒤 공백 제거)
df['AS '] = df['AS '].str.lower().str.strip()
filtered_df['AS '] = filtered_df['AS '].str.lower().str.strip()

# AS 라벨 분포 계산
as_distribution = filtered_df['AS '].value_counts().reset_index()
as_distribution.columns = ['AS_Label', 'Count']

# 전체 AS 라벨 분포 계산 (비교용)
all_as_distribution = df['AS '].value_counts().reset_index()
all_as_distribution.columns = ['AS_Label', 'Count']

# 결과 저장
with open('/home/psw/AS_Radiomics/data/dataprep/KMU_AS_CRF/AS_CRF_CT_ECG_summary.txt', 'w', encoding='utf-8-sig') as f:
    f.write("===== 분석 결과 =====\n\n")
    f.write(f"전체 환자 수: {len(df)}\n")
    f.write(f"모든 검사일자 간 간격이 1년 이내인 환자 수: {len(filtered_df)}\n\n")
    
    f.write("===== 모든 검사일자 간 간격이 1년 이내인 환자의 AS 라벨 분포 =====\n")
    for i, row in as_distribution.iterrows():
        f.write(f"{row['AS_Label']}: {row['Count']} ({row['Count']/len(filtered_df)*100:.2f}%)\n")
    
    f.write("\n===== 1년 이내 기준을 만족하는 환자 목록 =====\n")
    f.write("1차년도연구번호, Cardiac CT검사일자, Chest CT검사일자, Echocardiography_date, AS\n")
    for _, row in filtered_df.iterrows():
        cardiac_ct = row['Cardiac CT검사일자'].strftime("%Y-%m-%d") if pd.notnull(row['Cardiac CT검사일자']) else "NaT"
        chest_ct = row['Chest CT검사일자'].strftime("%Y-%m-%d") if pd.notnull(row['Chest CT검사일자']) else "NaT"
        echo_date = row['Echocardiography_date'].strftime("%Y-%m-%d") if pd.notnull(row['Echocardiography_date']) else "NaT"
        f.write(f"{row['1차년도연구번호']}, {cardiac_ct}, {chest_ct}, {echo_date}, {row['AS ']}\n")
    
    # 비교를 위해 전체 데이터의 AS 라벨 분포도 저장
    f.write("\n===== 전체 환자의 AS 라벨 분포 (비교용) =====\n")
    for i, row in all_as_distribution.iterrows():
        f.write(f"{row['AS_Label']}: {row['Count']} ({row['Count']/len(df)*100:.2f}%)\n")

# 콘솔에 요약 정보 출력
print(f"전체 환자 수: {len(df)}")
print(f"모든 검사일자 간 간격이 1년 이내인 환자 수: {len(filtered_df)}")
print("\n===== 모든 검사일자 간 간격이 1년 이내인 환자의 AS 라벨 분포 =====")
for i, row in as_distribution.iterrows():
    print(f"{row['AS_Label']}: {row['Count']} ({row['Count']/len(filtered_df)*100:.2f}%)")
    
print("\n분석 결과가 AS_CRF_CT_ECG_summary.txt 파일에 저장되었습니다.")
