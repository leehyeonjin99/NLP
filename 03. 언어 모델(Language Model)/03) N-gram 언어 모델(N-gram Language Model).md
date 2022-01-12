n-gram 언어 모델 : 카운트에 기반한 통계적 접근을 사용하는 SLM의 일종
- 이전에 등장한 모든 단어가 아닌 일부 단어만 고려하는 접근 방법
- n : 일부 단어를 몇 개보느냐 결정

# 1. 코퍼스에서 카운트하지 못하는 경우의 감소.
SLM의 한계
- 훈련 코퍼스에 확률을 계산하고 싶은 문장이나 단어가 없을 수 있다
- 확률을 계산하고 싶은 문장이 길어질수록 갖고있는 코퍼스에서 그 문자잉 존재하지 않을 가능성이 높다. : 카운트 하지 못할 가능성이 높다.

P(is|An adorable little boy) ≈ P(is|boy)  
P(is|An adorable little boy) ≈ P(is|little boy)  
갖고 있는 코퍼스에 An adorable little boy is가 있을 가능성보다 boy is 또는 little boy is라는 더 짧은 시퀀스가 존재할 가능성이 더 높다.
이제는 단어의 확률을 구하고자 기준 단어의 앞 단어를 전부 포함해서 카운트하는 것이 아니라, 앞 단어 중 임의의 개수만 포함해서 카운트하여 근사한다.   
→ 코퍼스에서 해당 단어의 시퀀스를 카운트할 확률 증가

# 2. N-gram
임의의 개수를 정하기 위한 기준을 위해 사용 n-gram 사용
n-gram : n개의 연속적인 단어 나열 의미   
An adorable little boy is spreading smiles
- **uni**gram : an, adorable, little, boy, is, spreading, smiles
- **bi**gram : an adorable, adorable little, little boy, boy is, is spreading, spreading smiles
- **tri**gram : an adorable little, adorable little boy, little boy is, boy is spreading, is spreading smiles
- **4**-gram : an adorable little boy, adorable little boy is, little boy is spreading, boy is spreading smiles

n-gram을 통한 언어 모델에서는 다음 나올 단어의 예측은 오직 n-1개의 단어에만 의존한다. 예를 들어 **'An adorable little boy is spreading smiles'** 다음에 나올 단어를 예측하고 싶다고 할 때, n=4라고 한 4-gram을 이용한 언어 모델을 사용한다고 하자.  
이 경우, spreading 다음에 올 단어를 예측하는 것은 n-1에 해당되는 앞의 3단어만을 고려한다.

![image](https://user-images.githubusercontent.com/57162812/149064703-82a9f7cc-cadf-442b-8a74-d092ebbf701e.png)

갖고 있는 코퍼스에서 boy is spreading가 1,000번 등장 : boy is spreading insults가 500번 등장, boy is spreading smiles 200qjs emdwkd
→ P(insults|boy si spreading)=0.5, P(smiles|boy is spreading)=0.2
→ 확률적 선책에 의하여 insults가 더 맞다고 판단

# 3. N-gram Language Model의 한계
앞의 4-gram 언어 모델은 주어진 문장에서 앞에 있던 **'An adorable little'** 이라는 수식어를 제거하고, 반영하지 않았다. **'작고 사랑스러운'** 수식어까지 모두 고려하여 **작고 사랑스러운 소년** 이 하는 행동에 대해 다음 단어의 예측하는 모델이었다면 **'모욕을 퍼트렸다'** 라는 부정적인 내용이 **'웃음을 지었다'** 라는 긍정적인 내용 대신 선택되었얼까?

n-gram은 뒤의 단어 몇 개만 보다 보니 의도하고 싶은 대로 문장을 끝맺음하지 못하는 경우가 생긴다.
→ 전체 문장을 고려한 언어 모델보다는 정확도가 떨어진다.

#### (1) 희소 문제(Sparsity Problem)
문장에 존재하는 앞에 나온 단어를 모두 보는 것보다 일부 단어만을 보는 것이 현실적으로 코퍼스에서 카운트 할 수 있는 확률을 높일 수 있지만, n-gram 언어 모델도 여전히 희소 문제 존재

#### (2) n을 선택하는 것은 trade-off 문제
임의의 개수인 n을 1보다는 2로 선택하는 것이 거의 대부분의 경우에서 언어 모델의 성능을 높일 수 있다. 가령, spreading만 보는 것보다는 is spreading을 보고 다음 언어를 예측하는 것이 더 정확하기 때문이다. 이 경우, 훈련 데이터가 적절한 데이터였다면 언어 모델이 적어도 spreading 다음에 동사를 고르지는 않을 것이다.

- n을 크게 선택한 경우 
   - 실제 훈련 코퍼스에서 해당 n-gram을 카운트할 수 있는 확률이 적어지므로 희소 문제는 점점 심각
   - 코퍼스의 모든 n-grma에 대해서 카운트 → 모델 사이즈 증가

- n을 작게 선택하는 경우
   - 근사의 정확도는 현실의 확률분포와 멀어진다.
   - trade-off 문제로 인해 정확도를 높이려면 **n은 최대 5를 넘게 잡아서는 안 된다고 권장** 되고 있다.

월스트리스트 저널에서 3,800만 개의 단어 토큰에 대하여 n-gram 언어 모델을 학습하고, 1,500만 개의 테스트 데이터에 대해서 테스트 했을 떄의 성능 : 펄플렉서티(perplexity)는 수치가 낮을수록 성능이 더 좋다.

![image](https://user-images.githubusercontent.com/57162812/149066404-012a2344-10a8-4081-968e-fb7b60c1fe1f.png)

n을 올릴 때마다 성능이 올라간다.

# 4. 적용 분야(Domain)에 맞는 코퍼스의 수집
분야, 어플리케이션에 따라 특정 단어들의 확률 분포는 상이  
- 마켓팅 분야 → 마켓팅 단어 빈번
- 의료 분야 → 의료 관련 단어 빈번
언어 모델에 사용하는 코퍼스를 해당 도멘인의 코퍼스를 사용한다면 당언히 언어 모델이 제대로된 언어 생성을 할 가능성이 높아진다.

때로는 이를 언어 모델의 약점이라고 하는 경우도 있다. : 훈련에 사용된 도메인 코퍼스가 무엇이냐에 따라서 성능이 비약적으로 달라진다.

# 5. 인공 신경망을 이용한 언어 모델(Neural Network Based Language Model)
N-gram Language Model의 한계점 극복  
→ (일반화 방법) 분모, 분자에 숫자를 더해서 카운트했을 때 0이 되는 것을 방지  
→ 완전한 해결 실패  
→ 대안 : **인공 신경망을 이용한 언어 모델**
