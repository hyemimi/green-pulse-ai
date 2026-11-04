# 모델 입출력 데이터 스키마

## 처리 구조

```text
1분 센서 데이터
  → 반응기별 결측 처리·이력 저장
  → 176개 시계열 특징 생성
  → [1단계 HGB] 정상/결함 확률과 경보
  → [2단계 Extra Trees] 매분 F1~F4 유형 확률
  → 대시보드 경보 단계·추정 유형
```

이 시스템은 이진 탐지에서 끝나지 않는 2단계 구조다. HGB는 결함 여부를 빠르게 탐지하고, 별도의 온라인 Extra Trees가 경보 이후 매분 F1~F4 유형 확률을 갱신한다. 자세한 연결 방식은 [2단계 모델 구조](./two_stage_model_architecture.md)를 참고한다.

대시보드는 176개 특징을 직접 전달하지 않는다. 원시 센서값을 전달하면 백엔드의 특징 생성기가 학습 때와 동일한 순서와 정의로 모델 입력을 만든다.

## 1. 원시 입력 스키마

데이터 주기는 반응기당 1분이다.

### 식별·운전 정보

| 필드 | 타입 | 필수 | 설명 |
|---|---|---:|---|
| `timestamp` | ISO 8601 datetime | Y | 센서 측정 시각 |
| `reactor_id` | string | Y | 반응기 식별자와 시계열 그룹 키 |
| `operating_regime` | enum(`A`,`B`) | Y | 현재 운전 조건 |

### 센서 측정값

| 필드 | 타입 | 필수 | 설명 |
|---|---|---:|---|
| `reactor_temp` | float/null | Y | 반응기 온도 |
| `reactor_pressure` | float/null | Y | 반응기 압력 |
| `feed_flow_rate` | float/null | Y | 원료 유량 |
| `coolant_flow_rate` | float/null | Y | 냉각수 유량 |
| `agitator_speed_rpm` | float/null | Y | 교반기 회전 속도 |
| `reaction_rate` | float/null | Y | 반응 속도 |
| `conversion_rate` | float/null | Y | 전환율 |
| `selectivity` | float/null | Y | 선택도 |
| `yield_pct` | float/null | Y | 수율 |
| `vibration_rms` | float/null | Y | 진동 RMS |
| `motor_current` | float/null | Y | 모터 전류 |
| `power_consumption_kw` | float/null | Y | 소비 전력 |

### 설정값

| 필드 | 타입 | 필수 | 설명 |
|---|---|---:|---|
| `temp_setpoint` | float/null | Y | 목표 온도 |
| `pressure_setpoint` | float/null | Y | 목표 압력 |

`null`은 센서 누락을 의미한다. 값이 없는 필드를 요청에서 제거하지 말고 명시적으로 `null`로 전달하는 편이 데이터 품질 추적에 유리하다.

### 입력 예시

```json
{
  "timestamp": "2026-11-04T10:31:00+09:00",
  "reactor_id": "A_R1",
  "operating_regime": "A",
  "reactor_temp": 81.4,
  "reactor_pressure": 5.21,
  "feed_flow_rate": 102.7,
  "coolant_flow_rate": 88.3,
  "agitator_speed_rpm": 1202.0,
  "reaction_rate": 0.73,
  "conversion_rate": 91.2,
  "selectivity": 87.5,
  "yield_pct": 79.8,
  "vibration_rms": 1.26,
  "motor_current": 31.7,
  "power_consumption_kw": 18.9,
  "temp_setpoint": 80.0,
  "pressure_setpoint": 5.0
}
```

## 2. 운영 입력에서 제외할 필드

다음 값은 학습 정답 또는 미래 정보이므로 모델 입력으로 받지 않는다.

- `fault_type`
- `is_fault`
- `time_to_fault_min`
- `efficiency_loss_pct`

운영 이후 실제 정비 결과가 확인되면 별도의 피드백 데이터로 저장할 수 있지만 추론 입력과는 분리해야 한다.

## 3. 결측 처리와 상태 저장

- 센서와 설정값은 같은 `reactor_id`의 마지막 관측값으로 최대 6분까지만 채운다.
- 다른 반응기의 값이나 미래값을 사용하지 않는다.
- 6분을 초과한 결측은 자동으로 채우지 않고 데이터 품질 이상으로 처리한다.
- 원래 누락 여부, 마지막 정상 수신 시각, 연속 누락 시간을 보존한다.

특징 생성을 위해 반응기별 이력이 필요하다.

| 이력 | 용도 |
|---|---|
| 최근 60분 | 변화량과 5·15·60분 이동 통계 |
| 최근 6시간 | 장기 기준 z-score와 15분 평균 편차 |
| 최근 24시간 확률 | 상대 변화 shadow 신호 |
| 최근 경보 상태 | persistence, hysteresis, cooldown, incident 병합 |

서비스 재시작 후에도 동일한 예측을 이어가려면 이 이력을 메모리에만 두지 않고 복구 가능한 저장소에 보존해야 한다.

## 4. 내부 모델 입력

HGB 입력은 설정 파일에 고정된 순서의 176개 float 특징이다.

### 기본 152개

- 현재 센서값 12개와 설정값 2개
- `regime_B`, `missing_sensor_count`
- 1·5·15·30·60분 변화량
- 5·15·60분 이동평균과 이동표준편차
- 온도·압력 설정값 오차
- 냉각수/원료 유량 비율
- 전력/전류 비율

### 추가 24개

12개 센서마다 다음 두 특징을 생성한다.

- `{sensor}__z_6h`: 현재값의 직전 6시간 기준 z-score
- `{sensor}__mean15_vs_6h`: 15분 평균의 직전 6시간 기준 편차

정확한 이름과 순서는 [binary_fault_detector_optimized_config.json](../models/binary_fault_detector_optimized_config.json)의 `feature_columns`를 단일 기준으로 사용한다.

## 5. 모델 직접 출력

HGB의 직접 출력은 해당 시점의 결함 확률 하나다.

| 필드 | 타입 | 범위 | 설명 |
|---|---|---|---|
| `fault_probability` | float | 0~1 | 현재 시점의 결함 확률 |

확률 하나만으로 사용자 경보를 발생시키지 않는다. 이전 분의 확률과 현재 경보 상태를 함께 사용한다.

## 6. 대시보드 출력 스키마

| 필드 | 타입 | 설명 |
|---|---|---|
| `timestamp` | ISO 8601 datetime | 판정 기준 시각 |
| `reactor_id` | string | 반응기 식별자 |
| `fault_probability` | float | 현재 HGB 결함 확률 |
| `alert_stage` | enum | `normal`, `signal`, `warning`, `confirmed`, `data_quality` |
| `threshold` | float/null | 현재 단계 진입 임계값 |
| `persistence_count` | integer | 연속 임계값 초과 횟수 |
| `incident_id` | string/null | 병합된 경보 사건 식별자 |
| `incident_started_at` | datetime/null | 현재 incident 시작 시각 |
| `predicted_fault_type` | integer/null | 추정 F1~F4 코드 |
| `predicted_fault_name` | string/null | 결함 유형 이름 |
| `fault_type_confidence` | float/null | 유형 분류 신뢰도 |
| `type_prediction_status` | enum | `unavailable`, `provisional`, `confirmed_candidate` |
| `reason_signals` | array | 주요 센서 방향과 기준선 편차 |
| `data_quality` | object | 누락·지연 센서 상태 |

### 출력 예시

```json
{
  "timestamp": "2026-11-04T10:32:00+09:00",
  "reactor_id": "A_R1",
  "fault_probability": 0.47,
  "alert_stage": "confirmed",
  "threshold": 0.45,
  "persistence_count": 2,
  "incident_id": "A_R1-20261104-0012",
  "incident_started_at": "2026-11-04T10:31:00+09:00",
  "predicted_fault_type": 1,
  "predicted_fault_name": "Cooling System Degradation",
  "fault_type_confidence": 0.38,
  "type_prediction_status": "provisional",
  "reason_signals": [
    {"sensor": "reactor_temp", "direction": "up", "baseline_deviation": 1.2},
    {"sensor": "coolant_flow_rate", "direction": "down", "baseline_deviation": -0.9}
  ],
  "data_quality": {
    "status": "good",
    "missing_sensor_count": 0,
    "stale_sensors": []
  }
}
```

## 7. 경보 단계 계산

| 단계 | 진입 조건 | 용도 |
|---|---|---|
| `signal` | 확률 0.225 이상 1분 | 화면 표시와 관찰 강화 |
| `warning` | 확률 0.250 이상 2분 | 작업자 확인 요청 |
| `confirmed` | 확률 0.450 이상 2분 | 대응 절차 시작 |

단계 해제는 확률이 한 번 낮아졌다고 즉시 수행하지 않는다. 일정 시간 정상 상태가 유지될 때 해제하는 hysteresis와 재알림을 억제하는 cooldown이 별도로 필요하다.

기존 최적화 모델의 개발 기준선은 `0.35 이상 2분`이지만, 대시보드 운영 표현에는 전체 정상 재생에서 교정한 다단계 기준을 사용한다. 두 설정을 혼용하지 않도록 설정 버전을 출력 데이터와 함께 기록하는 것이 좋다.

## 8. 결함 유형 출력

이진 HGB는 결함 여부만 판단한다. `predicted_fault_type`은 HGB 경보 이후 매분 실행되는 온라인 Extra Trees 유형 분류기의 출력이다.

- 경보 직후: 첫 F1~F4 확률과 `provisional` 유형
- 이후 매분: 새 센서값으로 유형 확률 갱신
- 경보 10분 후: Validation 기준 `confirmed_candidate`

Test는 경보 즉시 7/8, 10분 후 8/8이었지만 표본이 작고 확률이 낮아 자동 조치에는 사용하지 않는다. F1~F4 물리 규칙과 관계 특징은 `reason_signals`의 설명 정보에도 사용한다.

## 9. 구현 시 고정해야 할 버전 정보

예측 결과를 재현하려면 다음 버전을 함께 기록한다.

- 모델 파일 버전
- 특징 스키마 버전과 176개 컬럼 순서
- 결측 처리 정책 버전
- 경보 임계값과 persistence 버전
- hysteresis·cooldown 정책 버전
- 입력 데이터의 측정 단위와 센서 보정 버전

