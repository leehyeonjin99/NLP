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