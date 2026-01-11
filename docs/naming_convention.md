# Naming Convention Document (version 1.0)

### 원칙
- 파일명은 식별자(identifier)로 사용
- 의미/조건/판정은 메타 데이터(metadata, yaml)로 관리
- 네이밍 규칙은 보기 좋음 또는 편이성보다 안정성이 우선

#### 폴더명: 측정 세트 전체를 대표하는 고유 식별자
- 필드 구분자료 밑줄(`_`) 사용
- `{measured_date}_{product_code}_{description}` 날짜_제품코드_설명 → 하나의 논리적 단위
- 각 부분이 독립적인 의미를 가짐
- 상위 레벨 메타데이터

#### 파일명: 개별 측정 데이터의 파라미터
- 필드 구분자료 하이픈(`-`) 사용
- `{pattern}-{frequency}-{dimming}-{data_type}` 패턴-주파수-디밍 → 측정 조건
- 각 부분이 변수/파라미터
- 하위 레벨 데이터

### 기존 문제점 / 제약사항

- 현재: `{pattern} {frequency} {dimming}`
- 레거시 패턴명/파일명의 문제점: 구분자 (밑줄 및 공백 사용)
- 레거시 패턴명(밑줄 포함/밑줄로 끝나는 경우 포함)
- 공백 문제와 소수점 문제

```yaml
구분자 규칙:
  필드 구분자: "-" (하이픈)     # 절대 규칙
  내부 구분자: "_" (밑줄)       # 각 필드 내부에서만 허용
  소수점: "." (점)             # 숫자의 소수점만 허용
  확장자 구분: "_" (밑줄)       # 데이터 타입 구분

금지 사항:
  - 필드 구분에 밑줄 사용 금지
  - 파일명에 공백 사용 금지
  - 구분자 혼용 금지
```

## 폴더 구조 / 파일명 네이밍 규칙

### 1. Measured Product Identification (폴더명)

- 폴더: `{measured_date}_{product_code}_{description}` (밑줄 `_` 로 구분)
- 예시: `20251125_AA12345678_vertical_crosstalk_random_mura_2nd`
- `measured_date`: 측절 날짜 YYYYMMDD 8자리 (예시: 20251125)
- `product_code`: OLDE 제품 고유코드 10자리 (예시: AA12345678)
- `descroption`: 평가 목적 / 불량 / 개발 단계 / 평가자 등


```python
fonler_name = "20251125_AA12345678_vertical_crosstalk_random_mura_2nd"
parts = folder_name.split('_', 2)  # 최대 3개로 분리
measured_date = parts[0]    # "20251125"
product_code = parts[1]       # "AA12345678"
description = parts[2]      # "vertical_crosstalk_random_mura_2nd"
```

### 2. Measured/preprocessed Data Identification (파일명)

- 파일: `{pattern}-{frequency}-{dimming}-{data_type}` (하이픈 `-` 으로 구분)
- 예시: `t10_a_-60-200-xyz.npz` (평가 패턴 t10_a_, 주파수 60 Hz, White 휘도 200 nit, xyz 데이터)
- `pattern`: OLED Display 평가 패턴 (pattern00 ~ pattern99) 원본 RGB 이미지 이름
- `frequency`: OLED Display 평가 주파수 (정수 또는 실수)
- `dimming`: OLED Display 평가 패턴의 White 휘도값 (정수 또는 실수)
- `data_type`
  - `xyz`: 2D 계측기로 측정한 X (channel 1), Y (channel 2), Z (channel 3) raw 데이터 (`*.npz`)
  - `rgb`: XYZ data (`*.npz`) 를 변환한 RGB 이미지 `*.png`
  - `rgb_norm`: XYZ data (`*.npz`) 를 dimming (Y_white) 값으로 normalize 하여 변환한 RGB 이미지 (`*.png`)
  - `gray`: Y data (`*.npz` 파일의 channel 2) 를 변환한 gray 이미지 (`*.png`)
  - `gray_norm`: Y data (`*.npz` 파일의 channel 2) 를 normalize 하여 변환한 gray 이미지 (`*.png`)

### 3. Trained Model/config Identification (파일명)

- 파일: 
  - `weights_{구동화질}-{category}-{model}-{timestamp}.pth` (학습된 가중치 파일 저장)
  - `configs_{구동화질}-{category}-{model}-{timestamp}.yaml` (학습 조건 및 성능 저장)
  - `train_{구동화질}-{category}-{model}-{timestamp}.log` (학습 결과 및 성능 저장)
- `category`: 유사 패턴을 묶어 지정한 카테고리 이름 (카테고리별 이상감지 모델 학습)
- `model`: defecvad 에서 구현된 이상감지 모델 이름 (stfpm, efficientad, dinomaly)
- `timestamp`: 학습한 날짜 YYYYMMDD_hhmmss 15자리 (마지막 최신 파일 사용)

### 4. Inference Result Identification (파일명)

- 형식: `{pattern} {frequency} {dimming}-{category}-{model}`  (빈칸 ` `/`_` 으로 구분)
- 예시: `pattern16 60 200 category02 stfpm`
- `pattern`: OLED Display 평가 패턴 (pattern00 ~ pattern99)
- `frequency`: OLED Display 평가 주파수 (정수 또는 실수)
- `dimming`: OLED Display 평가 패턴의 White 휘도값 (정수 또는 실수)
- `category`: 유사 패턴을 묶어 지정한 카테고리 이름 (카테고리별 이상감지 모델 학습)
  - `category01`: `pattern00` ~ `pattern09` (cross talk)
  - `category02`: `pattern10` ~ `pattern19` (line defect)
  - `category03`: `pattern20` ~ `pattern29` (mura)
- `model`: defecvad 에서 구현된 이상감지 모델 이름


## 폴더 / 파일 구조

### 1. 측정 / 전처리 데이터

```
data_archive/
└── {measured_date}_{product_code}_{description}/
    ├── patterns
    │   └── {pattern}.png
    ├── measured
    │   └── {pattern}-{frequency}-{dimming}_xyz.npz
    └── preprocessed
        ├── {pattern}-{frequency}-{dimming}-rgb.png
        ├── {pattern}-{frequency}-{dimming}-rgb_norm.png
        ├── {pattern}-{frequency}-{dimming}-gray.png
        └── {pattern}-{frequency}-{dimming}-gray_norm.png
```

### 2. 학습된 모델

```
trained_models/
    └── {dataset}
        └── {category}
            └── {model}
                ├── weights-{category}-{model}-{timestamp}.pth
                ├── configs-{category}-{model}-{timestamp}.yaml
                └── train-{category}-{model}-{timestamp}.log
```

### 3. 추론 결과
```
ai_inspection/
└── {measured_date}_{product_code}_{description}/
    └── {category}
        └── {category}
            └── {model}
                └── {timestamp}
                    └── {pattern}-{frequency}-{dimming}-{category}-{model}_map.png
```
