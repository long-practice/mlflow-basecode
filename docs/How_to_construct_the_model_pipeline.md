## 모델 파이프라인 구성

### 모델 파이프라인 호출
모델 파이프라인은 `./main.py`에서 호출하며,<br>
이 때, 훈련 및 테스트 데이터, 하이퍼 파라미터 튜닝 간 적용할 `trial`수, test_data 분할 비율, `mlflow`사용 여부를 설정<br>
![image](https://github.com/long-practice/mlflow-basecode/assets/83870423/b05b7f50-b755-45d9-8346-ee1b02884045)


모델 파이프라인은 `./src/pipeline.py`에 구성

<br>

### 모델 파이프라인 구성
- 모델 파이프라인은 아래와 같이 구성
  - `preprocess`: 전처리 파이프라인을 호출하여 `self.train` 혹은 `self.test`를 전처리
  - `model_tuning`: 하이퍼 파라미터 튜닝을 위해 원하는 모델 생성, 모듈 호출
  - `run`: 모델 파이프라인 플로우를 구성 및 실행<br>
![image](https://github.com/long-practice/mlflow-basecode/assets/83870423/4bea1815-8d4f-461d-9272-7973c25ff87b)

- 하이퍼 파라미터 튜닝을 진행할 모델은 아래와 같이 선언<br>
  - 이 때 적용할 오차함수와 입력변수 X, 종속변수 y(훈련, 검증 데이터)를 정의
  - 또한 `optuna`라이브러리에서 `study`객체를 생성하여 하이퍼 파라미터 튜닝을 진행할 때 오차의 방향을 설정
  - 이 외 학습횟수, 로거, `mlflow`사용 여부 설정<br>
![image](https://github.com/long-practice/mlflow-basecode/assets/83870423/12dd9805-4916-4b76-bef4-9a8a2875083e)
