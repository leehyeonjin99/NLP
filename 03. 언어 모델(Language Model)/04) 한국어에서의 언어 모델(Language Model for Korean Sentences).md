# 1. 한국어는 어순이 중요하지 않다.
이전 단어가 주어졌을 때, 다음 단어가 나타날 확률을 구해야하는데 어순이 중요하지 않다는 것은 다음 단어로 어떤 단어든 등장할 수 있다는 의미

Ex)
- 나는 운동을 합니다 체육관에서.
- 나는 체육관에서 운동을 합니다.
- 체육관에서 운동을 합니다.
- 나는 운동을 체육관에서 합니다.

4개의 문장은 전부 의미가 통한다. 단어의 순서를 뒤죽박죽으로 바꾸어놔도 한국어의 의미가 전달되기 때문에 확률에 기반한 언어 모델이 제대로 다음 단어를 예측하기 어렵다.

# 2. 한국어는 교착어이다.
띄어쓰기 단어인 어절 단위로 토큰화를 할 경우에는 문장에서 발생가능한 단어의 수가 굉장히 늘어난다.  
예를 들어, 교착어인 한국어에는 어떤 행동을 하는 동사의 주어나 목적어를 위해서 조사가 있다.

'그녀'라는 단어 하나만 해도 그녀가, 그녀를, 그녀와, 그녀의 등과 같이 다양한 경우 존재  
→ **토큰화** 를 통해 접사나 조사 등을 분리하는 것은 중요한 작업

# 3. 한국어는 띄어쓰기가 제대로 지켜지지 않는다.
한국어는 띄어쓰기를 제대로 하지 않아도 의미가 전달, 띄어쓰기 규칙 또한 상대적으로 까다로운 언어  
→ 자연어 처리를 하는 것에 있어서 한국어 코퍼스는 띄어쓰기가 제대로 지켜지지 않는 경우 다수  
→ 토큰이 제대로 분리 되지 않은채 훈련 데이터로 사용된다면 언어 모델은 제대로 동작 불가능
