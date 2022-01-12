# 1. 조건부 확률
- P(A), P(B)에 대하여
   - P(B|A)=P(A⋂B)/P(A)  
   - P(A⋂B)=P(A)P(B|A)
- P(A),P(B),P(C),P(D)에 대하여
   - P(A⋂B⋂C⋂D)=P(A)P(B|A)P(C|A⋂B)P(D|A⋂B⋂C)
- 조건부 확률의 연쇄 법칙(chain rule) : n개에 대하여 일반화
![image](https://user-images.githubusercontent.com/57162812/149053249-1ad5991b-2acc-4802-95f0-60d1c009c721.png)

# 2. 문장에 대한 확률
P(An adorable little boy is spreading smiles) = 'An adorable little boy is spreading smiles'의 확률
![image](https://user-images.githubusercontent.com/57162812/149053655-8d8b6ef2-0675-4d24-920b-45632e44eb33.png)  
→ (적용) ![image](https://user-images.githubusercontent.com/57162812/149053755-0a1d0f6b-e143-44c5-b633-e32a6e13c40f.png)

# 3. 카운트 기반의 접근
이전 단어로부터 다음 단어에 대한 확률 구하는 방법 : 카운트 기반

![image](https://user-images.githubusercontent.com/57162812/149054056-bf662df2-2ead-4f5b-8a72-dcf05f4c9f77.png)  
예를 들어 기계가 학습한 코퍼스 데이터에서 An adorable little boy가 100번 등장했는데 그 다음에 is가 등장한 경우는 30번이라고 하자. 이 경우 P(is|An adorable little boy)는 30%이다.

# 4. 카운트 기반 접근의 한계 - 희소 문제(Sparsity Problem) 
언어 모델은 실생활에서 사용되는 언어의 확률 분포를 근사 모델링한다.
언어 모델의 목표 : 기계에 많은 코퍼스를 훈련시켜 언어 모델을 통해 현실에서의 확률 분포를 근사
→ 카운트 기반으로 접근시, 방대한 양의 훈련 데이터 필요  

![image](https://user-images.githubusercontent.com/57162812/149054056-bf662df2-2ead-4f5b-8a72-dcf05f4c9f77.png)  
- 훈련한 코퍼스에 An adorable little boy is 라는 단어 시퀀스가 없다 : 확률=0
- An adorable little boy라는 단어 시퀀스가 없다 : 분모=0 : 정의되지 않는 확률

**희소 문제(sparsity problem)** : 충분한 데이터를 관측하지 못하여 언어를 정확히 모델링하지 못하는 문제
- 완화 방안 : n-gram 언어 모델, 스무딩, 백오프
- 한계로 인하여 언어 모델의 트렌드는 SLM에서 인공 신경망 언어 모델로 넘어간다.
