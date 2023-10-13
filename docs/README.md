## mlflow-basecode
## Test
전처리 파이프라인, 모델 파이프 라인 확인 노트북 <br>
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
├── docker/
│   ├── build_docker_image.sh
│   └── Dockerfile
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
│   ├── logging.py
│   ├── model_tuning.py
│   └── parameter.py
├── docs/
│   └── README.md
├── create_container.sh
├── main.py
├── requirements.txt
└── run_main.sh
```

## How To Make Submission(for kaggle)
1. 전처리 파이프라인 구성: `preprocessing/preprocess.py`에서 구현([참고](https://github.com/long-practice/mlflow-basecode/blob/main/docs/How_to_construct_the_preprocessing_pipeline.md))
2. 모델 파이프라인 구성: `src/pipeline.py`에서 구현([참고](https://github.com/long-practice/mlflow-basecode/blob/main/docs/How_to_construct_the_model_pipeline.md))
3. `optuna`라이브러리 이용 하이퍼파라미터 튜닝, `utils.parameter.py`에서 각 모델, 변수별 타입, lower bound, upper bound 지정([참고](https://github.com/long-practice/mlflow-basecode/blob/main/docs/How_to_do_hyperparameter_tuning.md))
4. `log`파일에서 튜닝과정 확인, `artifacts`에 원하는 모델이 있는지 확인
5. 최종적으로 노트북에서 추가 작업 진행(앙상블, submission 생성 및 다운로드)

## How To Use MLflow
1. 전처리, 모델 파이프라인 수정
2. 모델 하이퍼 파라미터 튜닝 간 필요 패키지, 버전을 명시(`./requirements.txt` 수정)<br>
   해당 패키지들은 도커 컨테이너에서 파이프라인 실행하기 직전에 다운로드 받을 예정
   
   ![image](https://github.com/long-practice/mlflow-basecode/assets/83870423/8d52748d-37ba-4ed5-8440-455228440b1a)

3. Docker 이미지 생성
   ```
   /bin/bash ./docker/build_docker_image.sh
   ```
   Docker 이미지 확인
   ```
   docker images
   ```
   Docker 이미지 삭제
   ```
   docker rmi [IMAGE ID]
   ```

   ![image](https://github.com/long-practice/mlflow-basecode/assets/83870423/ffff3ed6-8284-4790-9ac5-69c3c833a363)

4. Docker 컨테이너 진입(상호 작용모드로 bash쉘 이용)<br>
   컨테이너 진입 전 산출물 경로 지정(반드시 절대경로로 명시)<br>
   반드시 `create_container.sh`에서 수정
   ```
   ARTIFACT_DIR=/.../mlflow-basecode/artifact/
   LOG_DIR=/Users/.../mlflow-basecode/logs/
   DATA_DIR=/.../mlflow-basecode/data/
   PREDICTION_DIR=/.../mlflow-basecode/prediction/
   ```
   ```
   /bin/bash ./create_container.sh
   ```
   
6. Docker 컨테이너 내부에서 메인 파일 실행 (필요 시 `nohup` 및 백그라운드 실행)
   ```
   /bin/bash ./run_main.sh
   ```
   
7. Docker 컨테이너 내부에서 mlflow ui 띄우기 (하이퍼 파라미터 튜닝 종료 후 실행)
   ```
   mlflow server -h 0.0.0.0
   ```
8. 로컬에서 웹으로 접속(`https://localhost:5000`)
9. 결과 확인
  - experiment 선택해서 각각의 run기록을 보거나 특정 run기록을 선택하여 하이퍼 파라미터 간 비교 가능(compare)
  - mlflow 서버 실행 정지: `ctrl + c`
  - 결과 확인 후 Docker 컨테이너 빠져나오기(Docker 컨테이너를 빠져나오면 컨테이너 자동 삭제): `exit`
