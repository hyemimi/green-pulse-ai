# 로컬 데이터 준비와 재생성

대용량 원본과 재생성 가능한 중간 산출물은 Git에 포함하지 않는다. 원본 CSV는 로컬에만 보관하고 코드·설정·소형 평가 결과·최종 모델만 버전 관리한다.

## 디렉터리 구조

```text
data/
├── chemical_process_timeseries.csv       # 로컬 원본, Git 제외
├── processed/                            # 전처리 결과, Git 제외
└── modeling/                             # 대용량 .csv.gz는 Git 제외
```

## 원본 데이터

원본을 다음 경로에 배치한다.

```text
data/chemical_process_timeseries.csv
```

이 저장소에는 원본 다운로드 URL이 기록되어 있지 않다. 원본을 별도로 보관하거나 최초 제공처에서 다시 받아야 한다.

검증된 원본 파일 정보:

| 항목 | 값 |
|---|---|
| 크기 | 209,843,962 bytes |
| SHA-256 | `1224d44e3e77e3d51c5ccc4b449f802c3f8a0d38a4ed7bda017539edbf039aba` |
| 분석 행 수 | 777,600 |
| 기간 | 2024-01-01 ~ 2024-03-30 |
| 반응기 | 6개 |

macOS/Linux에서 체크섬을 확인할 수 있다.

```bash
shasum -a 256 data/chemical_process_timeseries.csv
```

## 전처리

실시간 인과적 결측 처리를 실행한다.

```bash
python3 scripts/preprocess_realtime_missing.py
```

생성 파일:

```text
data/processed/chemical_process_timeseries_realtime_ffill.csv.gz
data/processed/chemical_process_timeseries_realtime_ffill.metadata.json
```

정책은 반응기별 최대 6분 forward fill이며 미래값과 정답 컬럼을 보간하지 않는다.

## 모델링 산출물 재생성 순서

노트북을 다음 순서로 실행한다.

```text
01 EDA
02 결측 분석
03~04 결함 전·후 패턴
05 실시간 특징과 이벤트 분할
06~12 이진 탐지·운영 정책·진단 실험
13 10분 탐지 가능성
14 HGB 경보 기준 집계형 유형 분류
15 온라인 유형 분류
```

대용량 산출물은 다음과 같이 재생성된다.

| 파일 | 생성 단계 |
|---|---|
| `data/modeling/realtime_event_features.csv.gz` | 노트북 05 |
| `data/modeling/event_split_manifest.csv` | 노트북 05 |
| `data/modeling/continuous_replay_predictions.csv.gz` | 노트북 10 |
| `data/modeling/test_binary_predictions*.csv.gz` | 노트북 06·09 |

## Git에 포함하는 산출물

- `docs/`, `notebooks/`, `scripts/`, `tools/`
- 모델 설정 JSON
- 특징 스키마와 소형 평가 CSV
- 최종 HGB와 온라인 유형 분류 모델

111MB의 이전 Extra Trees 기준 모델은 최종 파이프라인에서 사용하지 않으므로 제외한다.

## 주의사항

- `.gitignore`는 로컬 파일을 삭제하지 않고 Git 추적만 방지한다.
- 원본 파일은 저장소 밖에 별도 백업하는 것이 좋다.
- 현재 전처리 스크립트는 원본을 생성하거나 다운로드하지 않는다.
- 완전한 원클릭 재생성 파이프라인은 아직 없으며 노트북 순서 실행이 필요하다.

