## mlflow-basecode
## Test
<a href="https://colab.research.google.com/drive/1XabJj5QJzwwxP5GLmBwJiC38PtcH-bMm" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Directory

```
project-root/
├── artifact/
│   ├── LGBM_model.pkl
│   └── XGB_model.pkl
├── data/
│   ├── TRN.csv
│   └── test.csv
├── notebook/
│   ├── EDA_1.ipynb
│   ├── EDA_n.ipynb
│   └── predict.ipynb
├── preprocessing/
│   └── preprocess.py
├── prediction/
│   └── submission.csv
├── src/
│   ├── pipeline.py
│   └── styles/
│       └── notebook_style.css
├── utils/
│   ├── logging.csv
│   └── model_tuning.py
├── docs/
│   └── README.md
└── LICENSE.txt
```

## How To Use
1. 전처리 파이프라인 구성: `preprocessing/preprocess.py`에서 구현
2. 모델 파이프라인 구성: `src/pipeline.py`에서 구현
3. `optuna`라이브러리 이용 하이퍼파라미터 튜닝, `utils.parameter.py`에서 각 모델, 변수별 타입, lower bound, upper bound 지정
4. `log`파일에서 튜닝과정 확인, `artifacts`에 원하는 모델이 있는지 확인
5. 최종적으로 노트북에서 추가 작업 진행(앙상블, 제출)

