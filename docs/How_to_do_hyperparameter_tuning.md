## 하이퍼 파라미터 튜닝

하이퍼 파라미터 튜닝은 `./utils/model_tuning.py`에서 진행 <br>
그러나 모델 파이프라인에서 선언해준 모델에 대하여 튜닝할 파라미터들을 `utils.parameter.py`에서 관리

<br>

### 파라미터 지정
- 아래와 같이 모델별 튜닝할 파라미터의 이름, 하한값, 상한값, 데이터 타입을 설정(**범주형 변수는 개발중**)
![image](https://github.com/long-practice/mlflow-basecode/assets/83870423/c4ebf731-69e5-4637-ab65-2954065af917)
- `./utils/model_tuning.py`의 `get_params` 메소드 수정
  - 적용할 모델(정확히는 모델 파이프라인으로부터 넘겨받은 `self.model_name`에 대해  `utils.parameter.py`에서 파라미터 정보를 불러옴
  - 불러온 파라미터 정보들을 이용하여 `{parameter: trial_suggest_{data_type}(parameter_name, lower_bound, upper_bound)`형태로 딕셔너리 구성
  - 위의 LightGBM 파라미터 정보들에 의하면 다음과 같은 딕셔너리를 구성해야 함
    ```
    lgbm_params = {
        'verbosity': trial.suggest_int('verbosity', -1, -1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 10, 500),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
    }
    ```
