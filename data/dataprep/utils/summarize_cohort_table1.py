"""405 코호트의 인구학·연도별 등록·분할을 집계한다 — Table 1 과 Figure 1 의 숫자.

모델 산출물이 아니라 `AS_CRF.csv` 와 재분할 결과의 집계이므로 학습 없이 지금 돌아간다.
CRF 는 같은 환자를 연구번호 두 개로 중복 등록한 행이 있어 그 쌍을 먼저 찾아 낸다 — 표의 n 이 스캔 수인지 환자 수인지가 갈린다.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

COHORT_CSV = '/home/psw/AS_Radiomics/data/datasets/Dataset004_mix_KUDH0467rm/mask_source.csv'
CRF_CSV = '/home/psw/AS_Radiomics/data/AS_CRF.csv'
CLASS_ORDER = ['normal', 'nonsevere', 'severe']
CLASS_LABEL = {'normal': 'Normal', 'nonsevere': 'Non-severe', 'severe': 'Severe'}
SPLIT_LABEL = {'train': 'development', 'val': 'test'}
DAYS_PER_YEAR = 365.2425
RECENT_YEAR = 2024

# CRF 의 KUDH0910 행은 열이 밀려 CT 오더명과 검사일자가 성별·생년월일 자리에 들어갔다.
# 성별·나이·생년월일은 되살릴 값이 없고 검사일자만 명확해서 그것만 넣는다 — 없으면 연도별 행의 합이 404 가 된다.
CRF_OVERRIDES = {'KUDH0910': {'Cardiac CT검사일자': '2024-11-18',
                              '나이': np.nan, '생년월일': np.nan}}


class VerificationError(Exception):
    """코호트가 405 가 아니거나 CRF 조인이 깨지면 빈 표를 내는 대신 중단한다."""


def strip_dates(series):
    """CRF 날짜 열은 'x' 와 앞뒤 공백이 섞여 있어 벡터 파싱이 통째로 실패한다."""
    return pd.to_datetime(series.astype(str).str.strip(), errors='coerce')


def load_cohort(cohort_csv, crf_csv):
    """재분할 결과 405건에 CRF 를 붙인다. 조인 키는 연구번호이고 1:1 이어야 한다."""
    cohort = pd.read_csv(cohort_csv)[['patient_id', 'severity', 'split', 'has_gt']]
    if len(cohort) != 405:
        raise VerificationError(f"코호트가 405 건이 아니다: {len(cohort)}건")

    crf = pd.read_csv(crf_csv)
    crf = crf.rename(columns={crf.columns[0]: 'patient_id'})
    crf = crf[crf['patient_id'].notna()]  # 연구번호 칸이 빈 행이 580개 있어 조인 검증이 그냥은 안 선다
    for patient_id, fields in CRF_OVERRIDES.items():
        for column, value in fields.items():
            crf.loc[crf['patient_id'] == patient_id, column] = value

    df = cohort.merge(crf[['patient_id', '환자번호', '성별', '나이', '생년월일',
                           'Cardiac CT검사일자', 'Echocardiography_date']],
                      on='patient_id', how='left', validate='one_to_one')
    if df['환자번호'].isna().any():
        missing = df.loc[df['환자번호'].isna(), 'patient_id'].tolist()
        raise VerificationError(f"CRF 에 없는 환자: {missing}")

    df['sex'] = df['성별'].astype(str).str.strip().replace({'nan': np.nan})
    df['ct_date'] = strip_dates(df['Cardiac CT검사일자'])
    df['echo_date'] = strip_dates(df['Echocardiography_date'])
    df['birth_date'] = strip_dates(df['생년월일'])
    df['age_recorded'] = pd.to_numeric(df['나이'], errors='coerce')
    df['age_at_ct'] = (df['ct_date'] - df['birth_date']).dt.days / DAYS_PER_YEAR
    df['age'] = df['age_recorded'].fillna(df['age_at_ct'])
    df['year'] = df['ct_date'].dt.year
    df['ct_tte_days'] = (df['ct_date'] - df['echo_date']).dt.days.abs()
    return df


def find_duplicate_patients(df):
    """같은 병원등록번호로 두 번 등록된 쌍. 연구번호가 달라 분할이 이들을 한 사람으로 보지 못한다."""
    dup = df[df['환자번호'].duplicated(keep=False)].copy()
    return dup.sort_values(['환자번호', 'split', 'patient_id'])


def mean_sd(values):
    values = values.dropna()
    if len(values) < 2:
        return '—'
    return f"{values.mean():.1f} ± {values.std(ddof=1):.1f}"


def count_pct(group, sex):
    n = int((group['sex'] == sex).sum())
    return f"{n} ({n / len(group) * 100:.1f})"


def year_cell(group, year):
    rows = group[group['year'] == year]
    if not len(rows):
        return '0 (0/0)'
    male = int((rows['sex'] == 'M').sum())
    female = int((rows['sex'] == 'F').sum())
    return f"{len(rows)} ({male}/{female})"


def columns(df):
    """Table 1 의 열 순서 — Total 다음이 클래스 라벨 순서다."""
    return [('Total', df)] + [(CLASS_LABEL[cls], df[df['severity'] == cls]) for cls in CLASS_ORDER]


def print_table(df):
    """Table 1 의 본문 행을 그대로 낸다 — 손으로 옮기다 자릿수가 어긋나는 것을 막는다."""
    cols = columns(df)
    years = sorted(int(y) for y in df['year'].dropna().unique())

    def row(label, cell):
        print(f"| {label} | " + ' | '.join(cell(g) for _, g in cols) + ' |')

    print('\n=== Table 1 ===')
    print('| Variable | ' + ' | '.join(f"{name} (n={len(g)})" for name, g in cols) + ' |')
    print('| --- | --- | --- | --- | --- |')
    print('| **Demographics** | | | | |')
    row('Age (years)', lambda g: mean_sd(g['age']))
    row('Male Age', lambda g: mean_sd(g.loc[g['sex'] == 'M', 'age']))
    row('Female Age', lambda g: mean_sd(g.loc[g['sex'] == 'F', 'age']))
    print('| **Gender, n (%)** | | | | |')
    row('Male', lambda g: count_pct(g, 'M'))
    row('Female', lambda g: count_pct(g, 'F'))
    print('| **Yearly Enrollment, n (Male/Female)** | | | | |')
    for year in years:
        row(str(year), lambda g, y=year: year_cell(g, y))
    print('| **Dataset Partitioning, n** | | | | |')
    print('| *Classification Task* | | | | |')
    row('Development set', lambda g: str(int((g['split'] == 'train').sum())))
    row('Common held-out test set', lambda g: str(int((g['split'] == 'val').sum())))
    print('| *Segmentation Task* | | | | |')
    row('Development subset with AVC masks',
        lambda g: str(int(((g['split'] == 'train') & g['has_gt']).sum())))
    row('Common held-out test set with AVC masks',
        lambda g: str(int(((g['split'] == 'val') & g['has_gt']).sum())))


def print_ct_tte_interval(df):
    """본문 2.3 의 CT–TTE 간격. 표에는 안 들어가지만 같은 405 기준 재계산이다."""
    days = df['ct_tte_days'].dropna()
    q1, q3 = days.quantile([0.25, 0.75])
    print(f"\n=== CT–TTE 간격 (n={len(days)}) ===")
    print(f"  median {days.median():.0f} days (IQR {q1:.0f}–{q3:.0f}) · max {days.max():.0f} days")
    print(f"  같은 날 {int((days == 0).sum())}건 · 30일 이내 {int((days <= 30).sum())}건 "
          f"· 1년 초과 {int((days > 365).sum())}건")
    print(f"  echo 날짜 없음 {int(df['ct_tte_days'].isna().sum())}건")


def print_duplicates(dup):
    print(f"\n=== 병원등록번호가 겹치는 등록 ({len(dup)}행 · {dup['환자번호'].nunique()}명) ===")
    for hospital_id, pair in dup.groupby('환자번호'):
        splits = sorted(set(pair['split']))
        mark = ' ← development 와 test 를 가로지른다' if len(splits) > 1 else ''
        members = ' / '.join(f"{r.patient_id}({SPLIT_LABEL[r.split]}, {r.severity}, "
                             f"GT {'○' if r.has_gt else '✕'})" for r in pair.itertuples())
        ct_dates = sorted(set(pair['ct_date'].dt.strftime('%Y-%m-%d')))
        print(f"  {int(hospital_id)}: {members} · CT {', '.join(ct_dates)}{mark}")
    crossing = dup.groupby('환자번호')['split'].nunique()
    print(f"\n  가로지르는 쌍 {int((crossing > 1).sum())}쌍 · "
          f"중복 {len(dup) - dup['환자번호'].nunique()}건만큼 n 이 부풀어 있다")


def print_enrollment_drift(df):
    """연도별 등록이 클래스와 얽혔는지 — 얽혀 있으면 연도별 행이 프로토콜 표류를 라벨에 실어 나른다."""
    recent = df['year'] >= RECENT_YEAR
    table = pd.crosstab(df['severity'], recent)
    _, p_value, _, _ = chi2_contingency(table)
    print(f"\n=== 등록 연도와 클래스 ({RECENT_YEAR}년 이후 비율) ===")
    for cls in CLASS_ORDER:
        group = df[df['severity'] == cls]
        n_recent = int(recent[group.index].sum())
        print(f"  {CLASS_LABEL[cls]:>10}: {n_recent}/{len(group)} = {n_recent / len(group) * 100:.1f}%")
    print(f"  카이제곱 p = {p_value:.2e}")


def print_diagnostics(df):
    print('\n=== 결측 ===')
    for label, column in [('성별', 'sex'), ('나이(CRF 기록)', 'age_recorded'),
                          ('생년월일', 'birth_date'), ('CT 검사일자', 'ct_date'),
                          ('echo 검사일자', 'echo_date'), ('나이(보정 후)', 'age')]:
        missing = df.loc[df[column].isna(), 'patient_id'].tolist()
        print(f"  {label:>14}: {len(missing)}건" + (f" — {', '.join(missing)}" if 0 < len(missing) <= 5 else ''))

    both = df.dropna(subset=['age_recorded', 'age_at_ct'])
    gap = both['age_recorded'] - both['age_at_ct']
    print(f"\n=== 나이 두 정의의 차 (n={len(both)}) ===")
    print(f"  CRF 기록 나이 − CT 시점 나이: median {gap.median():+.2f}년 · "
          f"IQR {gap.quantile(0.25):+.2f}~{gap.quantile(0.75):+.2f} · 범위 {gap.min():+.2f}~{gap.max():+.2f}")
    print(f"  기록 나이만 쓰면 (n={int(df['age_recorded'].notna().sum())}) "
          f"mean {df['age_recorded'].mean():.2f} · 보정 후 (n={int(df['age'].notna().sum())}) "
          f"mean {df['age'].mean():.2f}")


def print_figure1_numbers(df, dup):
    """Figure 1 의 총계·제외 상자."""
    print('\n=== Figure 1 숫자 ===')
    print(f"  등록 스캔 {len(df)}건 · 고유 환자 {int(df['환자번호'].nunique())}명 "
          f"(중복 등록 {len(dup) - dup['환자번호'].nunique()}건)")
    for cls in CLASS_ORDER:
        group = df[df['severity'] == cls]
        print(f"  {CLASS_LABEL[cls]:>10}: {len(group)}건 "
              f"(development {int((group['split'] == 'train').sum())} · test {int((group['split'] == 'val').sum())})")
    print(f"  제외 1건 — KUDH0467 (조영증강 스캔, non-severe, development)")


def main():
    parser = argparse.ArgumentParser(description='405 코호트 집계 (Table 1 · Figure 1)')
    parser.add_argument('--cohort_csv', default=COHORT_CSV)
    parser.add_argument('--crf_csv', default=CRF_CSV)
    parser.add_argument('--out_csv', default=None, help='환자별 집계 입력값 저장 경로')
    args = parser.parse_args()

    print(f"코호트: {args.cohort_csv}")
    print(f"CRF   : {args.crf_csv}")

    df = load_cohort(args.cohort_csv, args.crf_csv)
    dup = find_duplicate_patients(df)
    print_table(df)
    print_ct_tte_interval(df)
    print_duplicates(dup)
    print_enrollment_drift(df)
    print_diagnostics(df)
    print_figure1_numbers(df, dup)

    if args.out_csv:
        keep = ['patient_id', '환자번호', 'severity', 'split', 'has_gt', 'sex', 'age_recorded',
                'age_at_ct', 'age', 'birth_date', 'ct_date', 'echo_date', 'year', 'ct_tte_days']
        df[keep].to_csv(args.out_csv, index=False)
        print(f"\n저장: {args.out_csv} ({len(df)}행)")


if __name__ == '__main__':
    main()
