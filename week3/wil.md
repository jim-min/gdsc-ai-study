자료의 출처는 대부분 여기.

https://wikidocs.net/book/2155



# 퍼셉트론

퍼셉트론이란 여러 변수와 가중치의 곱의 계산값에 편향값의 합으로 출력값을 뽑아내는 한 개의 과정을 의미합니다.

 

두뇌의 뉴런을 컴퓨터로 표현하기 위한 방식이고,

이러한 퍼셉트론을 다양한 층으로 이어 인공 신경망을 만들어낼 수 있습니다.

 

퍼셉트론의 한 단계에서 값을 어떻게 가공하는지 결정하는 것이 활성화 함수입니다.

(활성화 함수는 비선형 함수만 포함하는 말이긴 합니다)

 
![image](https://github.com/user-attachments/assets/aea7dbaf-9a78-4cec-b8b6-1270d21f2da2)


단순한 퍼셉트론은 이렇게 입력층과 출력층으로 나눠집니다.

 

## 단층 퍼셉트론
단층 퍼셉트론으로 우리는 간단한 AND, OR 게이트를 만들 수 있습니다.

 
```
def AND_gate(x1, x2):
    w1 = 0.5
    w2 = 0.5
    b = -0.7
    result = x1*w1 + x2*w2 + b
    if result <= 0:
        return 0
    else:
        return 1
 ```

단순하게 x1, x2이 둘 다 1이 아니면 0을 반환하는 함수입니다.

딱히 가중치값을 저렇게 해야만 하는 것도 아니고, 단순하게 하드코딩해서 만들 수 있습니다.

 
```
def OR_gate(x1, x2):
    w1 = 0.6
    w2 = 0.6
    b = -0.5
    result = x1*w1 + x2*w2 + b
    if result <= 0:
        return 0
    else:
        return 1
 ```

이건 반대로 x1, x2가 둘 다 0이 아니면 1을 반환하게 됩니다.

 

다만 XOR 게이트는 우리가 논리 회로 수업시간에 배울 수 있듯이 회로 하나로 만들 수 있는 것이 아니죠.

두 비트를 각각 AND와 NOR을 거치게 한 후, 두 결과값의 OR을 해야 합니다.

 

그러면 어떻게 해야 할 수 있을까요?

 

## 다층 퍼셉트론(MLP)
다층 퍼셉트론이라고 특별한 건 아닙니다. 퍼셉트론의 결과값을 또 퍼셉트론에 넣는 거라고 생각할 수 있습니다.

 

이렇게 입력층과 출력층 사이에 또 활성화 함수가 추가된 층을 은닉층이라고 부릅니다.

 

그렇다면, 은닉층의 두 노드에 각각 AND NOR을 넣고 OR 연산한 걸 출력하면 된다고 생각할 수 있습니다.

따로 구현을 하진 않겠습니다.

 

이러한 다층 퍼셉트론으로 텍스트를 분류하는 등 다양한 작업을 수행할 수 있습니다.

 

## 시그모이드 함수
활성화 함수로 사용되곤 하는 시그모이드 함수를 살펴보겠습니다.

 

뭐 단순합니다.
```
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
 ```

걍 이거고

![image](https://github.com/user-attachments/assets/842a94cd-f1ab-42a7-b89d-8e21e24dcf95)

 

이렇게 생겼습니다.

 

`matplotlib 시각화 코드`

```
import numpy as np
import matplotlib.pyplot as plt

# 시그모이드 함수 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 입력값 범위 설정
x = np.linspace(-10, 10, 100)  # -10부터 10까지 100개의 점
y = sigmoid(x)  # 시그모이드 적용

# 시각화
plt.plot(x, y, label='sigmoid')
plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.grid(True)
plt.legend()
plt.axhline(0, color='black', linewidth=0.5)  # x축
plt.axvline(0, color='black', linewidth=0.5)  # y축
plt.show()
```
볼 수 있듯이 값이 매우 커지거나 작아지면 미분값이 0 수준으로 매우 미미하여 가중치 적용이 잘 안 됩니다.

 

이렇기 때문에 시그모이드 함수는 역전파에서 사용을 지양하는 편입니다.

보통은 출력층에서만 사용하게 됩니다.

출력층에서 시그모이드 함수를 쓰면 이진 분류, 소프트맥스 함수를 쓰면 다중 분류, 회귀입니다.

 

tanh 함수를 쓰면 좀 덜하지만 여전히 비슷한 문제가 있고 보통 ReLU라는 함수를 사용하는 편입니다.

 

## ReLU 함수
렐루 함수는 0 이하면 0, 0 이상이면 x 값을 반환하는 간단한 함수입니다.

 
```
f(x)=max(0,x)
```
으로 표현 가능하며
```
def relu(x):
    return np.maximum(0, x)
 ```

코드 상에선 이렇습니다.

## 순전파와 역전파
순전파는 위 MLP와 비슷합니다.
입력층에서 은닉층을 거쳐 결과를 뽑아내는 방식입니다.

 

그렇다면 역전파는 갑자기 왜 하나요?

 

역전파는 순전파를 통해 나온 결과값과 실제값의 오차를 구하고\
그 오차를 바탕으로 하여 가중치, 편향값을 수정하는 과정입니다.

 

이 수정된 가중치로다가 다시 순전파를 돌리고,\
오차를 통해 가중치를 수정하는 과정을 거치면서\
모델의 학습이 이루어지는 것입니다.

 

우리가 머신러닝을 결과값 찾으려고 하는 게 아니고\
입력값을 바탕으로 최적의 결과를 뽑아내는 모델을 찾아내는 과정임을 곰곰히 생각해보면 됩니다.

 

역전파에서 미분값을 통하여 가중치를 어떻게 변화시킬지 결정하기 때문에,
기울기 소실 문제가 역전파에 악영향을 미치는 것입니다.

기울기가 거의 0이면 판단을 하기가 영 힘드니까요.\
반대로 오히려 필요없는 값을 0으로 제거해버리는 것이 ReLU 함수의 역할이기도 합니다.

 

## 제가 만든 예제

 

직접 모델을 만들어보겠습니다~
 ![image](https://github.com/user-attachments/assets/9bdd0906-3ea3-4556-8898-b0da04730a8d)

이러한.. 데이터를 통하여 품종을 맞추는 분류 문제입니다.

알코올에 든 성분을 통해 어떤 품종의 과일을 썼는지 맞추는? 그런 데이터인 것 같아요.

 

아직 다중 분류는 잘 몰라서 bce를 사용한 이진 분류만 해보겠습니다.

0이거나 1,2 이거나
```
import numpy as np
import pandas as pd
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from scipy.special import expit

alcohols_raw = pd.read_excel('./ex_exercise.xlsx', header=1)

alcohols_label = alcohols_raw.iloc[:, 0].to_numpy()
alcohols_feature = alcohols_raw.iloc[:, 21:34].to_numpy()
scaler = MinMaxScaler()
alcohols = scaler.fit_transform(alcohols_feature)
 ```

pd.read_excel(path, header = 1) 으로 해주면 1번 행은 열값으로 쓰고 다음 행부터 시작합니다.

1번째 행이 분류명이어서 이렇게 해줬습니다.

 

레이블 값과 피쳐 값 분류해주고 MixMaxScaler()로 값을 0~1 사이로 맞춰줍니다.

 
```
W = 2 * np.random.random((13,1)) - 1
y = (alcohols_label == 1).astype(int).reshape(-1, 1)
b = 0

Z = alcohols.dot(W) + b
```

가중치 랜덤, 편향값 0으로 시작해줍니다.

 
```
def bce_loss(y,y_hat):
    # log 0는 계산될 수 없기 때문에, 매우 작은 값으로 clip해줌
    minval = 0.000000000001
    N = y.shape[0]
    loss = -np.mean(y * np.log(y_hat.clip(min=minval)) + (1 - y) * np.log((1-y_hat).clip(min=minval)))
    return loss
 ```

Binary Cross Entropy 오차함수를 정의해줍니다.

로그에 0이 들어가면 에러가 나니까 min=minval으로 문제가 생기는 걸 방지해줍니다.
```
alpha = 0.02
epochs = 800

losses = []

for i in range(epochs):
    z = alcohols.dot(W) + b
    A = expit(z) # 시그모이드 함수

    # 손실 계산
    loss = bce_loss(y, A)
    print('Epoch:', i, 'Loss:', loss)
    losses.append(loss)

    dz = A - y
    dW = np.dot(alcohols.T, dz) / alcohols.shape[0]
    db = np.mean(np.sum(dz, axis=0, keepdims=True))

    W -= alpha * dW
    b -= alpha * db
 ```

시그모이드 함수는 그냥 import해와서 쓰고 800번 수행해줍니다.

이렇게
![image](https://github.com/user-attachments/assets/2d68eb90-bb7d-4f94-9806-356f165f74b3)


잘 나오는 것을 볼 수 있습니다!
