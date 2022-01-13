# 1. 선형 회귀(Linear Regression)
어떤 변수의 값에 따라서 특정 변수의 값이 영향을 받는다. 다른 변수의 값을 변하게 하는 변수를 x, 변수 x에 의해서 값이 종속적으로 변하는 변수 y라고 하자.  
선형 회귀 : 한 개 이상의 독립 변수 x와 y의 선형 관계를 모델링

#### 1) 단순 선형 회귀 분석(Simple Linear Regression Analysis)

![image](https://user-images.githubusercontent.com/57162812/149325131-e49343b3-ba30-41ad-ac67-96b64fd8c70e.png)

독립 변수 x와 곱해지는 값 w를 머신 러닝에서는 가중치(weight), 별도로 더해지는 b를 편향(bias)이라고 한다.

#### 2) 다중 선형 회귀 분석(Multiple Linear Regression Analysis)

![image](https://user-images.githubusercontent.com/57162812/149325334-ef5ed6f8-5459-4405-9a31-ea709ec3fb0d.png)

여러 개의 독립 변수 x와 종속 변수 y의 선형 관계

# 2. 가설(Hypothesis) 세우기
**단순 선형 회귀**를 가지고 문제를 풀어보자.

|hours(x)|score(y))
|:---:|:---:|
|2|25|
|3|50|
|4|42|
|5|61|

![image](https://user-images.githubusercontent.com/57162812/149325646-6af558b2-5186-4fcd-a7d8-11fd0a1f3226.png)

알고있는 데이터로부터 x와 y의 관계를 유추하고, 이 학생이 6시간, 7시간, 8시간을 공부하였을 때의 성적을 예측해보고 싶다.  
**가설(Hypothesis)** : x와 y의 관계를 유추하기 위해서 수학적으로 세운 식

![image](https://user-images.githubusercontent.com/57162812/149325873-c68047f5-39e7-44ed-80e0-5c049e879dc5.png)

위의 그림은 w와 b의 값에 따라서 천차만별로 그려지는 직선의 모습이다. 결국 선형 회귀는 주어진 데이터로부터 y와 x의 관계를 가장 잘 나타내는 직선을 그리는 일이다. 그리고 어떤 직선인지 결정하는 것은 w와 b의 값으로 선형 회귀에서 해야할 일은 결국 적절한 w와 b를 찾아내는 일이다.

# 3. 비용 함수(Cost function) : 평균 제곱 오차(MSE)
**목적 함수(Objective function) 또는 비용 함수(Cost function) 또는 손실 함수(Loss function)** : 실제값과 예측값에 대한 오차에 대한 식  
목적 함수 : 함수의 값을 최소화하거나 최대화하는 목적을 가진 함수 
비용 함수 또는 손실 함수 : 값을 최소화하려는 목적을 가진 함수

회귀 문제의 경우에는 주로 평균 제곱 오차(Mean Squared Error, MSE)를 사용

y와 x의 관계를 가장 잘 나타낸 직선을 그린다는 것 = 모든 점들과 위치적으로 가장 가까운 직선을 그린다는 것  
오차 : 주어진 데이터에서 각 x에서의 실제값 y와 예측하고 있는 H(x)값의 차이

오차의 크기를 측정하기 위한 가장 기본적인 방법은 모두 더하는 방법이다. y=13x+1 직선이 예측한 예측값을 각각 실제값으로부터 오차를 계산하여 표를 만들어 보자.

|hours(x)|2|3|4|5|
|:---:|:---:|:---:|:---:|:---:|
|실제값|25|50|42|61|
|예측값|27|40|53|66|
|오차|-2|10|-9|-5|

모든 오차를 더하면 음수 오차도 있고, 양수 오차도 있으므로 오차의 절대적인 크기를 구할 수 없다. → 모든 오차를 제곱해서 더한 후 평균을 구하는 방법 : MSE  
![image](https://user-images.githubusercontent.com/57162812/149328512-e2a65de9-8c95-4f3e-9d20-813da5104f05.png)
 
비용함수를 재정의해보면 다음과 같다.  
![image](https://user-images.githubusercontent.com/57162812/149328632-142ffbe1-58de-480d-9acb-9c0012cf7604.png)

 Cost(w, b)를 최소가 되게 만드는 w와 b를 구하면 결과적으로 y와 x를 가장 잘 나타내는 직선을 구할 수 있다.
 ![image](https://user-images.githubusercontent.com/57162812/149328928-fd888cdd-6d14-4ba0-bfe9-99b6b92f1a9a.png)
 
 # 4. 옵티마이저(Optimizer) : 경사하강법(Gradient Descent)
**옵티마이저(Optimizer) 또는 최적화 알고리즘** : 비용 함수를 최적화하는 매개 변수인 w와 b를 찾기 우히나 작업에 사용되는 알고리즘  
가장 기본적인 옵티마이저 알고리즘은 경사 하강법(Gradient Descent)이다.


(가정) 편향 b가 없이 단순히 가중치 w만을 사용한 y-wx라는 가설 H(x)를 가지고, 경사 하강법을 수행
w와 cost(w)의 관계를 그래프로 표현하자.

![image](https://user-images.githubusercontent.com/57162812/149329669-fd2c48ca-11a7-4a3d-b064-3d88921d2540.png)

기계가 해야할 일은 cost가 가장 최소값을 가지게 하는 w를 찾는 일이므로, 볼록한 맨 아래 부분의 w의 값을 찾아야 한다. 

![image](https://user-images.githubusercontent.com/57162812/149329998-182f1217-b79c-4f9e-b0d8-a989509b3128.png)

기계는 임의의 랜덤값 w값을 정한 뒤에, 맨 아래의 볼록한 부분(접선의 기울기가 0이 되는 지점)을 향해 점차 w의 값을 수정해나간다. 위의 그림은 w값이 점차 수정되는 과정이다. 이를 가능하게 하는 것이 경사 하강법이다.
   
아래의 식은 접선의 기울기가 음수이거나, 양수일 때 모두 접선의 기울기가 0인 방향으로 w값을 조정한다.
학습률(알파) : w의 값을 변경할 때, 얼마나 크게 변경할지 결정, 0과 1사이의 값
![image](https://user-images.githubusercontent.com/57162812/149356252-bb93f318-b4b6-48bb-9b8d-64a9718dc36d.png)

풀고자하는 각 문제에 따라 가설, 비용 함수, 옵티마이저는 전부 다를 수 있으며 선형 회귀에 가장 적합한 비용 함수와 옵티마이저가 알려져있는데 MSE와 경사하강법이 각각 이에 해당된다.


