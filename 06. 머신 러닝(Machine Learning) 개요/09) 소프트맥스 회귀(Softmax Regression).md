# 1. 다중 클래스 분류(Multi-class Classification)

시그모이드 함수 : 입력된 데이터에 대해서 0과 1 사이의 값을 출력하여 둘 중 하나에 속할 확률로 해석  
ex) 확률값이 0.5를 넘으면 스팸 메일로 판단, 아니면 정상 메일로 판단

- 이진 분류 : 두 개의 선택지 중 하나를 고르는 문제 
- 다중 클래스 분류 : 세 개 이상의 선택지 중 하나를 고르는 문제

ex) 꽃받침 길이, 꽃받침 넓이, 꽃잎 길이, 꽃잎 넓이로부터 setosa, versicolor, virginica라는 3개의 품목 중 어떤 품목인지 예측하는 문제

|SepalLengthCm(x1)|SepalWidthCm(x2)|PetalLengthCm(x3)|PetalWidthCm(x4)|Species(y)|
|:---:|:---:|:---:|:---:|:---:|
|5.1|3.5|1.4|0.2|setosa|
|4.9|3.0|1.4|0.2|setosa|
|5.8|2.6|4.0|1.2|versicolor|
|6.7|3.0|5.2|2.3|virginica|
|5.6|2.8|4.9|2.0|virginica|

만약 시그모이드 함수를 사용한다면? 각 정답지에 대해서 시그모이드 함수를 적용 : setosa가 정답일 확률 0.8, versicolor가 정답일 확률 0.2, virginica가 정답일 확률 0.4 → 전체 확률의 합계가 1이 아니다.

샘플 데이터가 들어오면 모델이 setosa일 확률이 0.7, versicolor일 확률이 0.05, virginica일 확률이 0.25와 같이 세 개의 총 합이 1인 예측값을 얻도록 하자. 이 경우 확률이 제일 높은 setosa로 예측한 것으로 간주하고자 한다. : 이럴 때 사용하는 것이 **소프트맥스 함수**이다.

# 2. 소프트맥스 함수(Softmax function)
소프트맥스 함수 : 선택해야 하는 선택지의 총 개수를 k라고 할 때, k차원의 벡터를 입력받아 각 클래스에 대한 확률 추정

#### 1) 소프트맥스 함수의 이해

k차원의 벡터에서 i번쨰 원소를 z_i, i번쨰 클래스가 정답일 확률을 p_i(0<=p_i<=1)로 나타낸다고 하자.  
소프트맥스 함수는 p_i를 다음과 같이 정의한다.

![image](https://user-images.githubusercontent.com/57162812/149523978-f95a93ae-ef37-416c-a34d-58386a09498f.png)

위의 경우 k=3, z=[z_1, z_2, z_3]

![image](https://user-images.githubusercontent.com/57162812/149524296-c09dc601-1d56-4069-a831-914529679e97.png)

#### 2) 그림을 통한 이해

![image](https://user-images.githubusercontent.com/57162812/149524482-555abc78-ddff-4b7c-9cfc-5ba179a9b19e.png)

- 4차원 벡터를 입력으로 받지만, 소프트맥스 함수의 입력으로 사용되는 벡터는 벡터의 차원이 분류하고자 하는 클래스의 개수가 되어야 하므로 어떤 가중치 연산을 통해 3차원 벡터로 변환되어야 한다.

샘플 데이터 벡터를 소프트 맥스 함수의 입력 벡터로 차원 축소하는 방법 : 소프트맥스 함수의 입력 벡터 z의 차원 수만큼 결과값이 가중치 곱을 진행

![image](https://user-images.githubusercontent.com/57162812/149524966-ad03e2ca-fc39-4dc8-b499-198f9cae74aa.png)

위의 그림에서 화살표는 총 12개이며 전부 다른 가중치를 가지고, 학습 과정에서 점차적으로 오차를 최소화하는 가중치로 값이 변경된다.

- 오차 계산 방법 : 소프트 맥스 함수의 출력은 클래스의 개수만큼 차원을 가지는 벡터, 각 원소는 0과 1 사이의 값을 가진다.

예측값과 비교를 할 수 있는 실제 값의 표현 방법 : 원-핫 벡터로 표현

![image](https://user-images.githubusercontent.com/57162812/149527356-2485fe8b-8f20-497a-a698-08163d66e909.png)

두 벡터의 오차를 계산하기 위해서 소프트맥스 회귀는 비용 함수로 크로스 엔트로피 함수를 사용

![image](https://user-images.githubusercontent.com/57162812/149527784-bcb5b100-8148-4180-8b08-19ed40ffa334.png)

![image](https://user-images.githubusercontent.com/57162812/149527855-e4730f07-5654-4083-aae7-5df2a73c46f0.png)

# 3. 원-핫 벡터의 무작위성

직관적으로 생각해볼 수 있는 레이블링 방법은 분류해야 할 클래스 전체에 정수 인코딩을 하는 것이다. 하지만 일반적으로 다중 클래스 분류 문제에서 레이블링 방법으로는 정수 인코딩이 아니라 원-핫  인코딩을 사용하는 것이 보다 클래스의 성질을 잘 표현하였다고 할 수 있다. 

(가정) Banana, Tomato, Apple라는 3개의 클래스가 존재하는 문제가 있다. 레이블은 정수 인코딩을 사용하여 각각 1, 2, 3을 부여하였다.  
손실 함수로 MSE를 사용하여보자.

실제값이 Tomato일 때 예측값이 Banana이었다면 제곱 오차는 (2-1)^2=1  
실제값이 Apple일 때 예츠갑이 Banana이었다면 제곱 오차는 (3-1)^2=4

즉, Banana과 Tomato 사이의 오차보다 Banana과 Apple의 오차가 더 크다. : 하지만, 정수 인코딩과 달리 원-핫 인코딩은 분류 문제 모든 클래스 간의 관계를 균등하게 분배한다.

원-핫 인코딩을 통한 레이블 인코딩에 따른 SE : 제곱 오차가 균등함을 확인 가능

![image](https://user-images.githubusercontent.com/57162812/149530614-d99a95af-20ce-4a94-b471-e514dc4a1ae7.png)

# 4. 비용 함수(Cost function)

#### 1) 크로스 엔트로피 함수

![image](https://user-images.githubusercontent.com/57162812/149531145-b4253088-d7cd-4384-8943-debea661235e.png)

c : 실제값 원-핫 벡터에서 1을 가진 원소의 인덱스, p_c=1 : 예측값이 y를 정확하게 예측한 경우  
→ 에측값이 y를 정확하게 예측한 경우 크로스 엔트로피 함수의 값은 0이 된다.

n개의 전체 데이터에 대한 평균을 구한다고 하면 최종 비용 함수는 다음과 같다.

![image](https://user-images.githubusercontent.com/57162812/149531660-2254882d-2e69-49c6-b0b5-3210a2d33e08.png)


#### 2) 이진 분류에서의 크로스 엔트로피 함수

![image](https://user-images.githubusercontent.com/57162812/149531807-04cbd558-c6df-476e-9c81-3f62291a5d74.png)  :  크로스 엔트로피의 함수식

y=y_1, 1-y=y_2로 치환, H(X)=p_1, 1-H(X)=p_2

![image](https://user-images.githubusercontent.com/57162812/149532064-5ae1c147-aa65-48a3-abd3-ee7c6a617a45.png)

![image](https://user-images.githubusercontent.com/57162812/149532100-a8d01bf2-9346-4052-968b-d1c127048489.png)

소프트맥스 회귀에서는 k의 값이 고정된 값이 아니므로 2를 k로 변경

![image](https://user-images.githubusercontent.com/57162812/149532206-42e55eb0-d456-4b57-827a-fb618322508b.png)

정리하면 소프트맥스 함수의 최종 비용 함수에서 k가 2라고 가정하면 결국 로지스틱 회귀의 비용 함수와 같다.\

![image](https://user-images.githubusercontent.com/57162812/149532327-7bef834c-a0ef-4143-9eb7-40e0002d4792.png)


# 5. 인공 신경망 다이어그램

![image](https://user-images.githubusercontent.com/57162812/149532378-e434fece-3ce8-4731-bcf3-ac70605ee488.png)

소프트맥스 회귀 또한 하나의 인공 신경망으로 볼 수 있다.


