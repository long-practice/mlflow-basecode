## 전처리 파이프라인 구성

### 전처리 파이프라인 호출
전처리 파이프라인은 `./src/pipeline.py`에서 `run`메소드에서 호출<br>
![image](https://github.com/long-practice/mlflow-basecode/assets/83870423/233b38ed-d277-469d-a950-5b66858f0181)

<br>

### 전처리 파이프라인 구성
- 전처리 파이프라인 다음과 같이 구성
  - `set_missing_value`: 결측치 채워넣는 메소드
  - `add_feature`: 파생변수 추가 혹은 피처 엔지니어링
  - `remove_outlier`: 이상치 제거
  - `do_preprocess`: 전처리 플로우 구성
- 이외 필요한 전처리에 대해 메소드들을 추가하거나 따로 모듈을 구성하여 구현할 수 있음<br>
![image](https://github.com/long-practice/mlflow-basecode/assets/83870423/21fb1775-3c2c-4dd5-9c54-26f70c5edf8d)
