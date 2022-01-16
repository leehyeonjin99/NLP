- **기울기 소실(Gradient Vanishing)** : 인공 신경망을 학습하다보면 역전파 과정에서 입력층으로  갈 수록 기울기가 점차적으로 작아지는 현상이 발생할 수 있어, 입력층에 가까운 층들에서 가중치들이 업데이트가 제대로 되지 않아 최적의 모델을 찾을 수 없게 되는 현상
- **기울기 폭주(Gradient Exploding)** : 기울기가 접ㅁ차 커지더니 가중치들이 비정상적으로 큰 값이 되면서 발산하는 현상

# ReLU와 ReLU 변형들

시그모이드 함수를 사용하면 입력의 절대값이 클 경우 시그모이드 함수의 출력값이 0또는 1에 수렴하면서 기울기가 0에 가까워진다. → 기울기 소실 문제 발생

기울기 소실 완화 방법 : ReLu나 Leaky ReLU 사용

- 은닉층에 시그모이드 함수를 사용하지 말기
- Leaky ReLU를 사용하면 모든 입력값에 대해서 기울기가 0에 수렴하지 않아 죽은 ReLu 문제 해결
- 은닉층에서 ReLU나 Leaky ReLU를 사용하자.

# 2. 그래디언트 클리핑(Gradient Clipping)

- 그래디언트 클리핑 : 기울기 폭주를 막기 위해 임계값을 넘지 않도록 임계치만큼 크기를 감소시키는 것

```python
from tensorflow.keras import optimizers

Adam=optimizer.Adam(lr=0.0001, clipnorm=1.)
```

# 3. 가중치 초기화(Weight initialization)

#### 1) 세이비어 초기화(Xavier Initialization)
- **세이비어 초기화(Xavier Initialization)** : 세이비어 글로럿과 요슈아 벤치오가 가중치 초기화가 모델에 미치는 영향을 분석하여 제안한 새로운 초기화 방법  
- 균등 분포 또는 정규 분포로 초기화 할 때 두 가지 경우가 있다.
- 이전 층의 뉴런 개수(n_in)와 다음 층의 뉴런 개수(n_out)로 이루어진 식

(균등 분포 사용 가중치 초기화) ![image](https://user-images.githubusercontent.com/57162812/149661430-72734416-3d94-42d7-b6ac-c73b5141036a.png)

(정규 분포 사용 가중치 초기화) 평균 0, 표준 편차 σ  
![image](https://user-images.githubusercontent.com/57162812/149661483-c6c3b04c-541f-4eb0-bcad-c9994dbb4cf8.png)

- S자 형태의 활성화 함수(시그모이드 함수, 하이퍼볼릭 탄젠트 함수)와 함께 사용할 경우 좋은 성능
- ReLU 함수 또는 ReLU의 변형 함수들을 활성화 함수로 사용할 경우에는 다른 초기화 방법을 사용하는 것이 탁월 : He 초기화(He initialization)

#### 2) He 초기화(He initialization)

- 다음 층의 뉴런의 수를 반영하지 않는다.

(균등 분포 초기화) ![image](https://user-images.githubusercontent.com/57162812/149661853-d2a77a4b-92ed-4e7b-ae1e-4f9b7a3ef941.png)

(정규 분포 초기화) ![image](https://user-images.githubusercontent.com/57162812/149661863-808c13e7-443c-4a0c-a58c-c48967ee0303.png)

- 시그모이드 함수나 하이퍼보릭탄젠트 함수를 사용할 경우에는 세이비어 초기화 방법이 효율적
- ReLU 계열 함수를 사용할 경우에는 He 초기화 방법 효율적
- ReLU + He 초기화 바업ㅂ이 좀 더 보편적

# 4. 배치 정규화(Batch Normalization)

배치 정규화는 인공 신경망의 각 층에 들어가는 입력을 평균과 분산으로 정규화하여 학습을 효율적으로 만든다.

#### 1) 내부 공변량 변화(Internal Covariate Shift)

- 내부 공변량 변화 : 학습 과정에서 **층 별로 입력 데이터 분포가 달라지는 현상**
- 공변량 변화는 훈련 데이터의 분포와 테스트 데이터의 분포가 다른 경우를 의미
- 내부 공변량 변화는 신경망 층에서 발생하는 입력 데이터 분포 변화를 의미

#### 2) 배치 정규화(Batch Normalization)

- 배치 정규화 : 한 번에 들어오는 배치 단위로 정규화
- 각 층에서 활성화 함수를 통과하기 전에 수행
- (정리) 입력에 대해 평균을 0으로 만들고, 정규화 → 정규화 된 데이터에 대해서 스케일과 시프트 수행 → γ : 스케일 관련 매개변수. β : 시프트 관련 매개변수

![image](https://user-images.githubusercontent.com/57162812/149662243-ecd0e58c-7896-49fa-9fbd-114b00ba8957.png)

#### 3) 배치 정규화의 한계

##### 1. 미니 배치 크기에 의존적이다.
##### 2. RNN에 적용하기 어렵다.

# 5. 층 정규화(Layer Normalization)

(배치 정규화) 미니 배치 : 동일한 특성 개수들을 가진 다수의 샘플들

![image](https://user-images.githubusercontent.com/57162812/149662397-27269822-d463-4383-a64b-ef5d61a9caa1.png)


(층 정규화)

![image](https://user-images.githubusercontent.com/57162812/149662410-5cb263e5-51ba-4979-903e-68e0d2e87ea9.png)


