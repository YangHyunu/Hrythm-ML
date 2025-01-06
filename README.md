# Hrythm-ML
![image](https://github.com/user-attachments/assets/6ac2d17a-82d5-4051-98e0-5da54c3fdb18)
![image](https://github.com/user-attachments/assets/6c12f682-dcb0-4aae-8562-5393447edc95)

---

## 프로젝트 개요
2024년도 2학기 **AI를 위한 머신러닝 팀프로젝트**로, 팀 **Hrythm**이 참여한 프로젝트입니다.  
이번 프로젝트는 Analytics Vidhya 플랫폼에서 진행 중인 **HR Analytics Practice Problem** 대회를 기반으로 진행되었습니다.  
머신러닝 기반 예측 모델을 통해 **승진 대상자 예측** 문제를 해결하고, 기업의 승진 프로세스를 효율화하는 것을 목표로 합니다.

---

## 데이터 및 대회 정보
- **대회 플랫폼**: [Analytics Vidhya](https://www.analyticsvidhya.com/)  
- **대회명**: HR Analytics Practice Problem  
- 본 프로젝트는 **WNS Analytics Wizard 2018** 대회의 데이터를 사용하였습니다.  

### 주요 차이점
1. **WNS Analytics Wizard 2018**: 평가지표로 **Accuracy** 사용.
2. **현 대회**: 평가지표로 **F1-Score**를 중점적으로 사용하여 **클래스 불균형 문제**를 해결하고자 함.

---

## 인사 데이터의 특징 및 평가지표 선택의 중요성

### 1. 클래스 불균형 문제
- 일반적으로 **인사 데이터**에서 승진 대상자(1)와 비승진 대상자(0)의 비율은 매우 불균형합니다.  
- 승진 대상자는 전체 데이터에서 극히 일부에 불과하며, 이를 제대로 예측하지 못하면 모델의 활용 가치는 떨어집니다.  

### 2. Recall(재현율)의 중요성
- 승진 대상자를 놓치지 않는 것이 중요하기 때문에 **Recall(재현율)**을 높이는 것이 핵심입니다.  
- Recall이 낮으면 승진 대상자를 식별하지 못해 기업의 승진 프로세스에 부정적인 영향을 미칠 수 있습니다.

### 3. F1-Score의 의미
- **F1-Score**는 Precision(정밀도)과 Recall(재현율)의 조화를 평가하는 지표로, 클래스 불균형 문제에서 중요한 역할을 합니다.  
- 이번 대회에서 F1-Score를 평가지표로 사용하는 것은 **실제 승진 대상자 예측 문제에서의 의미**를 반영합니다.  
  - Precision과 Recall 간의 균형을 유지하면서 승진 대상자를 놓치지 않는 성능을 최적화할 수 있습니다.

---

## 팀 Hrythm
- **팀 구성원 및 역할**:
  - **양현우**: 데이터 파이프라인설계/ 하이퍼파라미터 튜닝/ Stacking
  - **유르겐**: 데이터 전처리 / LDA,QDA,Logit_model
  - **박혜원**: EDA/자료조사/ppt
  - **우단비**: RandomForest, SVM /ppt
  - **김진원**: 자료조사
- **팀 목표**: 데이터 기반으로 효율적이고 신뢰성 있는 승진 예측 모델 개발.

---

## 주요 프로젝트 내용
### 1. 문제 정의
- 기업의 승진 프로세스를 효율화하기 위해 승진 가능성이 높은 직원을 사전에 예측.

### 2. 데이터 분석 및 모델링
- **데이터 활용**: 훈련 성과, KPI 완수율 등 데이터를 기반으로 **이진 분류 모델** 구축.
- **클래스 불균형 문제 해결**:
  - **오버샘플링** 및 **클래스 가중치 조정**으로 데이터 균형 조정.
- **모델 성능 개선**:
  - 여러 모델을 결합한 **스태킹**을 통해 성능 향상.
  - **오토인코더**를 활용해 피처를 추출하고 차원 축소 수행.

### 3. 결과 및 성과
- **최종 결과**: **157/17,737** 순위로 약 상위 0.88%에 해당하는 성적 달성. MLP 와 단일 베깅 모델에 비해 약 9%이상 f1 스코어가 증가함.
- **성과 및 배운 점**:
  1. **LDA, QDA** 등 통계적 학습 기반 모델 재탐구.
  2. 데이터 마이닝 기법(오버샘플링, 스태킹, 차원 축소)의 중요성 확인.
  3. 매우 불균형한 데이터에서는 단순 오버샘플링만으로 해결이 어렵고, 피처 엔지니어링과 오토인코더를 활용한 **특성 추출** 및 **스태킹**을 통해 성능이 크게 향상된다는 것을 확인.

---

## 기대 효과
- **프로세스 개선**: 승진 대상자 발표 지연을 줄이고 효율적인 전환 가능.
- **데이터 기반 의사결정**: 승진 대상자를 데이터와 모델 기반으로 신뢰성 있게 예측.
- **F1-Score 최적화**: 클래스 불균형 상황에서 신뢰성 있는 예측 성과 제공.

---

팀 Hrythm은 Analytics Vidhya 대회를 통해 머신러닝을 사용하여, 실제 비즈니스 문제를 해결하는 데 의미 있는 기여를 목표로 했습니다.


