![image](https://github.com/user-attachments/assets/4d287a29-c673-442a-a586-e99e42db2cb6)
# HR Analytics Practice Problem  

## 1. 프로젝트 개요
- **배경** : 중앙대학교 2024 2학기 AI를 위한 머신러닝 캡스톤 
- **기간**: 2024.11.08 ~ 2024.12.03 (1개월)  
- **인원**: 팀 Hrythm (5명)
  
| 이름    | 역할 및 기여                                    |
|---------|-------------------------------------------------|
| 양현우  | 데이터 파이프라인 설계, 하이퍼파라미터 튜닝, Stacking |
| 유르겐  | 데이터 전처리, LDA/QDA/Logit 모델 개발              |
| 박혜원  | EDA, 자료 조사, PPT 제작                         |
| 우단비  | RandomForest, SVM, PPT 제작                      |
| 김진원  | 자료 조사                                      |

- **플랫폼**: Analytics Vidhya  
- **데이터 출처**: WNS Analytics Wizard 2018 HR 데이터  
- **목표**:  
  머신러닝 기반으로 승진 대상자 예측 모델을 개발하고, 공정하고 효율적인 승진 프로세스 수립에 기여할 수 있는 인사이트를 도출.

## 🎯 문제 정의

- 기존 승진 예측은 관리자 주관에 의존해 **편향(bias)** 이 개입될 수 있음
- 데이터 기반 예측 모델로 **공정하고 일관된 승진 판단 근거**를 제시하고자 함
  
##  역할 및 기여

- 전처리 및 파이프라인 설계: 결측치 처리, 범주형 인코딩 자동화, 도메인 기반 파생 변수 설계
- RFECV 를 활용한 변수 선택
- Optuna 기반 하이퍼파라미터 튜닝 시스템 설계 및 구현
- XGBoost, LGBM, CatBoost, LDA 기반 스태킹 앙상블 구조 설계 및 검증
---

## 2. 데이터 이해 및 주요 이슈

- **데이터 특성**
  - 총 54,000여 명의 직원 데이터, 다중 범주형 및 연속형 변수 혼재
  - **승진자 비율 약 11%**로,  클래스 불균형 존재
  - 변수 간 다중공선성 및 타겟과의 낮은 선형 상관성으로 SMOTE 등 샘플링 기법 사용
  

- **분석 주요 이슈**
  - 전통적인 **상관분석(Pearson)** 및 **PCA** 적용 결과 타겟과 강한 선형 연관성 미약
  - 여러 파생 변수를 도출했으나 예측 성능에 큰 영향을 미치지 못함
  - 클래스 불균형을 해결하기 위해 SMOTE를 적용했으나, 단일 모델의 경우 타겟 특성이외의 특성들 또한 불균형이 심해 과적합이 발생하였음
  - → **비선형 모델 및 앙상블 구조 필요성 대두**

### 시각화

![image](https://github.com/user-attachments/assets/39be2097-f2a2-4aa8-b9ac-f8302c931665)

**PCA 결과** : 2개의 주성분으로 약 85% 이상의 총변동을 설명할 수 있었으나
  - 각 주성분에서의 중요 성분들이 'age', 'length_of_service', 'average_training_score' 등으로, 타겟과의 상관관계가 낮음
  - 대회 이후 사후적으로 오토인코더를 통해 비선형 특성을 추출한 후 모델을 학습하여 성능이 상당히 향상되었음

![image](https://github.com/user-attachments/assets/235a3622-ed20-4801-a449-44684a319818)
*변수 간 낮은 상관계수: 단일 변수 기반 예측의 한계*

---

## 3. 전처리 및 피처 엔지니어링

- 결측치 처리 및 범주형 변수 Encoding (One-hot, Label)
- 'education', 'length_of_service'에 결측치가 존재하여 각각 IterativeImputer, SimpleImputer로 처리
- `education`, `department`, `age`, `service_year` 등 도메인 기반 파생 변수 생성
  
**RFECV, Stepwise Feature Selection**으로 각 모델에 맞는 최적의 피처를 선택하였음

- **RFECV:**  
  - Recursive Feature Elimination with Cross-Validation을 통해 최적 피처 선택  
  - 최적 피처 수를 찾기 위해 5-fold CV를 적용
  - 부스팅과 배깅모델에만 적용
    
![image](https://github.com/user-attachments/assets/6c78114a-84ce-4986-b303-8a04ace2195c)

  
- **Stepwise Selection:**  
  - 전진선택법과 후진선택법을 통해 최적 피처 선택  
  - AIC/BIC를 기준으로 최적 피처 수를 찾음
![image](https://github.com/user-attachments/assets/be691644-49b4-4dbf-911e-7c8f7e6dd165)

<details>
  <summary><strong>전처리 파이프라인 코드 보기</strong></summary>

```python
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 전처리 파이프라인 클래스 정의
class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputer_rating = IterativeImputer(max_iter=10, random_state=42)
        self.imputer_education = SimpleImputer(strategy="most_frequent")
        self.scaler = StandardScaler()
        self.education_map = {'Below Secondary': 0, "Bachelor's": 1, "Master's & above": 2}

    def fit(self, X, y=None):
        self.imputer_rating.fit(X[['previous_year_rating']])
        self.imputer_education.fit(X[['education']])
        return self

    def transform(self, X):
        X[['previous_year_rating']] = self.imputer_rating.transform(X[['previous_year_rating']])
        X[['education']] = self.imputer_education.transform(X[['education']])
        X['education'] = X['education'].map(self.education_map)
        X['region'] = X['region'].str.replace('region_', '').astype(int)
        X.rename(columns={
            "employee_id": "id",
            "KPIs_met >80%": "KPIs_over_0.8",
            "awards_won?": "awards_won"
        }, inplace=True)
        for column in X.columns:
            if column in object_columns:
                dummies = pd.get_dummies(X[column], drop_first=True, prefix=f"{column}")
                dummies = dummies.astype(int)
                X = pd.concat([X, dummies], axis=1)
                X.drop(columns=column, inplace=True)
        return X

# 파이프라인 생성 함수
def create_pipeline():
    pipeline = Pipeline([
        ('preprocessor', Preprocessor())
    ])
    return pipeline

# 전처리 실행
pipeline = create_pipeline()
train = pipeline.fit_transform(train)
test = pipeline.transform(test)
```
</details>

### 3.1 하이퍼파라미터 튜닝
- **Optuna**를 사용하여 여러 ML 모델(lda,qda,rf,logit,LGBM, XGB, CatBoost, RF 등)을 개별 학습 후 평가했음. 여러 모델의 조합을 시도하여 앙상블 및 스태킹 적용
<details>
  <summary><strong>Optuna 하이퍼파라미터 튜닝 코드 보기</strong></summary>

<br>

```python
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC


class OptunaOptimizer:
    def __init__(self, models, train_X, train_y, metric, rfe=None):
        self.models = models
        self.train_X = train_X
        self.train_y = train_y
        self.metric = metric
        self.rfe = rfe if rfe else {model_name: train_X.columns.tolist() for model_name in models}
        self.studies = {}
        self.best_models = {}
        self.scaler = StandardScaler()

    def objective(self, trial, model_class, model_name):
        selected_features = self.rfe[model_name]
        train_X_filtered = self.train_X[selected_features]

        if model_class == LogisticRegression:
            param = {
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'C': trial.suggest_float('C', 1e-4, 1e4, log=True),
                'solver': trial.suggest_categorical('solver', ['saga']),
                'l1_ratio': trial.suggest_float('l1_ratio', 0, 1, log=False)
            }
        elif model_class == XGBClassifier:
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
            }
        elif model_class == LGBMClassifier:
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
            }
        elif model_class == RandomForestClassifier:
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }
        elif model_class == LinearDiscriminantAnalysis:
            solver = trial.suggest_categorical('solver', ['svd', 'lsqr', 'eigen'])
            param = {'solver': solver}
            if solver in ['lsqr', 'eigen']:
                param['shrinkage'] = trial.suggest_categorical('shrinkage', ['auto', None])
                if param['shrinkage'] is None:
                    param['shrinkage'] = trial.suggest_float('shrinkage_float', 0.0, 1.0)
        elif model_class == QuadraticDiscriminantAnalysis:
            param = {
                'reg_param': trial.suggest_float('reg_param', 0.0, 1.0)
            }
        elif model_class == SVC:
            kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
            param = {
                'C': trial.suggest_float('C', 1e-4, 1e4, log=True),
                'kernel': kernel,
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
            }
            if kernel == 'poly':
                param['degree'] = trial.suggest_int('degree', 2, 5)
                param['coef0'] = trial.suggest_float('coef0', 0.0, 1.0)
            if kernel == 'sigmoid':
                param['coef0'] = trial.suggest_float('coef0', 0.0, 1.0)
        elif model_class == CatBoostClassifier:
            param = {
                'iterations': trial.suggest_int('iterations', 100, 500),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-1, 1e2, log=True),
                'border_count': trial.suggest_int('border_count', 1, 255),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
                'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
                'od_wait': trial.suggest_int('od_wait', 10, 50)
            }

        train_split_X, val_X, train_split_y, val_y = train_test_split(train_X_filtered, self.train_y, test_size=0.2, random_state=42, stratify=self.train_y)
        train_split_X = self.scaler.fit_transform(train_split_X)
        val_X = self.scaler.transform(val_X)
        model = model_class(**param)
        model.fit(train_split_X, train_split_y)

        pred_y = model.predict(val_X)
        if self.metric == 'f1_score':
            score = f1_score(val_y, pred_y, average='macro')
        elif self.metric == 'accuracy_score':
            score = accuracy_score(val_y, pred_y)
        elif self.metric == 'roc_auc_score':
            score = roc_auc_score(val_y, model.predict_proba(val_X)[:, 1])
        return score

    def optimize(self, n_trials=100):
        for model_name, model_class in self.models.items():
            print(f"Starting hyperparameter tuning for {model_name}")
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: self.objective(trial, model_class, model_name), n_trials=n_trials)
            self.studies[model_name] = study
            print(f"Best parameters for {model_name}: {study.best_params}")
            print(f"Best {self.metric} for {model_name}: {study.best_value}")
            print(study)
            
    def train_best_models(self):
        for model_name, model_class in self.models.items():
            selected_features = self.rfe[model_name]
            train_X_filtered = self.train_X[selected_features]
            best_params = self.studies[model_name].best_params
            best_model = model_class(**best_params)
            
            # 데이터 스케일링
            train_split_X, val_X, train_split_y, val_y = train_test_split(train_X_filtered, self.train_y, test_size=0.2, random_state=42, stratify=self.train_y)
            train_split_X = self.scaler.fit_transform(train_split_X)
            val_X = self.scaler.transform(val_X)
            
            best_model.fit(train_split_X, train_split_y)
            self.best_models[model_name] = best_model

            pred_y = best_model.predict(val_X)
            f1 = f1_score(val_y, pred_y, average='binary')
            print(f"F1 score with best parameters for {model_name}: {f1}")


# Optimizer 인스턴스 생성 및 최적화 수행
optimizer = OptunaOptimizer(models, train_X, train_y, 'f1_score', rfe)
optimizer.optimize(n_trials=30)
optimizer.train_best_models()
```
</details>

---

## 4. 모델링 전략 및 성능 비교

### 4.1 단일 모델 기반 

| Model            | Val (SMOTE) | Test (비공개) | 성능 감소폭 |
|------------------|-------------|----------------|--------------|
| **LDA**          | 0.7200      | 0.4457         | ↓ 0.2743     |
| **QDA**          | 0.6806      | 0.4000         | ↓ 0.2806     |
| **Logistic Reg.**| 0.6158      | 0.3086         | ↓ 0.3072     |
| **KNN**          | 0.6200      | 0.2813         | ↓ 0.3387     |
| **Naive Bayes**  | 0.6100      | 0.2880         | ↓ 0.3220     |
| **SVM**          | 0.7400      | 0.2900         | ↓ 0.4500     |
| **Random Forest**| 0.9900      | 0.4200         | ↓ 0.5700     |

![image](https://github.com/user-attachments/assets/b785a596-f996-472f-8ea7-74542dd376d3)

- **훈련-검증 간 괴리** 현상 → 과적합 발생
- **비공개 테스트 세트**에서 F1-score 0.45 내외
  - 부스팅 모델이 가장 좋은 성능을 보였으나 역시 validation 에서는 0.7~ 0.8의 f1-score를 보여주며 훈련데이터와 테스트(비공개)데이터가 상당히 다른 분포를 가지고 있음을 알 수 있다.
---

### 4.2 Soft / Hard Voting 결과
- 부스팅과 배깅 모델이 그 자체로 보팅모델인 것을 알지만, 다양한 특성을 학습한 모델들을 바탕으로 보팅모델을 만들어 시도
![image](https://github.com/user-attachments/assets/b328f3ac-9c56-4d39-9f3c-2ba2edc7b4bd)

| Voting Method | Models | F1-Score |
|---------------|--------|----------|
| **Soft Voting** | xgb, lgbm, cat, lda | **0.4884** |
| Soft Voting | xgb, lgbm, rf, cat, lda, qda, logit | 0.4852 |
| Hard Voting | xgb, lgbm, cat, lda | 0.4795 |
| Hard Voting | xgb, lgbm, rf, cat, lda, qda, logit | 0.4442 |

> Soft Voting이 상대적으로 안정적인 성능 확보
    - Hard Voting의 경우  xgb, lgbm, cat과 같은 부스팅 모델들이 공통적으로 과적합하여 가장 좋지 못한 성능을 보였다.
따라서 앙상블 모델을 만들기 위해서는 다양한 특성을 학습한 모델들을 조합하여 앙상블 모델을 만들고, 그 결과를 바탕으로 스태킹 앙상블을 적용하기로 했다.


### 4.3 Stacking Ensemble 결과
> Stacking Ensemble  
  - 위 모델 학습 결과, 부스팅 계열이 아닌 모델들은 과소적합 또는 과적합 문제를 보였으며, 각 모델이 학습하는 피처들이 서로 다름을 확인할 수 있었음.  
이러한 점을 고려하여, 서로 다른 특성을 학습한 여러 모델의 예측 결과를 결합하는 스태킹 앙상블 방식을 사용하면 단일 모델보다 더 강력하고 일반화된 예측 성능을 얻을 수 있다고 판단함.  
즉, 각 베이스 모델이 제공하는 다양한 예측 정보를 메타 모델이 종합함으로써, 개별 모델의 약점을 보완하고 데이터 분포 차이 및 과적합 문제를 줄일 수 있을 것으로 기대함.

![image](https://github.com/user-attachments/assets/53d95b54-29e1-40a0-8fdb-5ef5e77ee455)


![image](https://github.com/user-attachments/assets/dcd6242a-bafe-4d3b-b91d-2788e16c5813)

> 메타 러너로 **LDA 사용 시 최고 성능**을 기록  
  - 본 결과에서는 단순 회귀 모형이나 단순 모델 대신 강력한 부스팅 계열 모델(XGB, LGBM, CatBoost 등)을 베이스 모델로 사용하고,  
  메타 모델로 선형 기반의 LDA를 적용하였을 때 최고 성능을 달성함.  
이는 부스팅 모델들이 학습한 다양한 비선형 특성을 효과적으로 결합한 후, LDA가 이를 선형적으로 조합하여 예측 성능을 극대화한 결과로 해석할 수 있음.  
특히, 일반적으로 메타 러너로 로지스틱 회귀 모델을 많이 사용하지만, 이번 결과에서는 LDA가 가장 우수한 성능을 보였으며,  
메타 러너로 QDA를 사용한 경우에는 베이스 모델들의 하이퍼파라미터 튜닝에 따라 성능 차이가 크게 나타남. 대회 사후 오토인코더를 통해 비선형 특성을 추출한 모델에서 최고 성능을 기록함.

---



## 5. 결과 및 인사이트
![image](https://github.com/user-attachments/assets/dd0c9fcc-7cb2-4264-bc14-d8fd1f182fb5)

- 핵심 변수: `kpi_met > 80%`, `awards_won`, `department`  
- 상대적으로 비중 낮은 변수: `length_of_service`, `education`, `no_of_trainings`

- **과적합**보다 **데이터 분포 차이**가 더 큰 성능 저하 요인이 될 수 있음
- 다양한 분포를 학습한 모델 조합과 **스태킹 전략**이 일반화 성능 확보에 핵심
- 단순한 모델 튜닝보다 **구조적 설계의 중요성**을 체감
- 향후에는 군집 기반 분석이나 비지도 학습을 결합한 **데이터 그룹별 학습 전략**이 필요

---

## 6. 회고 및 개선점

- 프로젝트 초기에는 주로 모델 학습, 전처리, 하이퍼파라미터 튜닝 등 **성능 최적화 중심의 기술적 기여**에 집중하였음.
- 그러나 이로 인해 데이터 기반 EDA를 통해 **인사 관련 인사이트 도출이나 핵심 변수 분석**은 상대적으로 부족한 결과를 초래함.
- 모델 성능은 일정 수준 이상 확보하였으나, **실제 인사 전략 수립에 활용 가능한 해석 중심 분석이 미흡**하다는 점을 사후에 인지하게 되었음.
- 이에 따라 프로젝트 종료 후 다음과 같은 보완을 진행함:
  - PCA 기반 차원 축소 기법의 한계를 극복하기 위해 **AutoEncoder를 활용한 비선형 잠재 특성 추출** 수행
  - **Kibana 기반 시각화 대시보드**를 구축하여 변수별 영향력과 분포를 직관적으로 파악하고 조직 차원의 활용 가능성을 확인
- 이 과정을 통해 단순한 예측 성능을 넘어서, **실제 비즈니스 의사결정에 기여하는 데이터 해석 역량**의 중요성을 체감함.

---


## 🛠 기술 스택

- Python, Scikit-learn, Pandas, NumPy  
- XGBoost, LightGBM, CatBoost, RF, QDA, LDA  
- SMOTE, Optuna, RFECV  
- Matplotlib, Seaborn

---

## 📎 참고 자료
- 📂 [사후 대시보드 및 EDA 보기](./대시보드_EDA.pdf)
- 📂 [최종 보고서 PDF 보기](./HR_Project.pdf)
- 📂 [최종 코드](./last_version.ipynb)
