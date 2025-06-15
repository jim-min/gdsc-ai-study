# 목차
- [1. SGD(Stochastic Gradient Descent)](#SGD(Stochastic-Gradient-Descent))
- [2. Adam](#adam)
- [3. Dropout](#dropout)
- [4. FNN(Feed-forward Neural Network)](#FNN(Feed-forward-Neural-Network))


# SGD(Stochastic Gradient Descent)
원래 배치 경사하강법은 데이터 전체에 대해 따져보고 역전파를 진행했었는데요,

이게 문제가 데이터 배치 하나를 통으로 살펴보다 보니 데이터 사이즈가 커질 수록 시간이 무지막지하게 오래 걸립니다.

 

그래서 SGD, Stochastic Gradient Descent를 사용하게 되는데요.\
전체 Train 세트를 쓰는 게 아니고 랜덤하게 미니 배치만 사용하게 되어 한 단계 당 걸리는 시간이 단축되게 됩니다!\
또한 Local minimum 문제에서 조금 더 자유롭다는 장점이 있어 최저점에 도달하기 더 쉬워집니다.

 

다만 이 경우에 노이즈가 조금 더 크고 더 많은 단계가 필요하다는 장점이 있습니다.\
물론이겠죠? 데이터를 다 쓰는 게 아니다보니...

요즘에는 더 나은 옵티마이저들이 많이 개발되어 사용되고 있습니다.

 

# Adam
그 중 하나인 Adam에 대해 살펴보겠습니다.

 
```
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

history= model.fit(x_train, y_train,
                   batch_size=32, epochs=10,
                   validation_data=(x_test, y_test))
 ```

이런 식으로 케라스 코드에서도 많이 쓰는 옵티마이저인데요,

 

Adam은 ADAptive Moment estimation을 줄인 것으로,  SGD에서 발달한 AdaGrad 알고리즘에서\
모멘텀 및 RMSProp을 결합하여 경사 하강법의 효율성을 크게 향상시킨 옵티마이저입니다.

 

참고로 RMSProp은 파라미터 변화량을 지수 가중 평균을 사용하여 스텝 사이즈를 최적화하는 알고리즘입니다.

일단 모르겠으면 Adam 박고 보라네요.\
스텝의 방향과 크기를 잘 최적화해준다고 합니다.

NAdam, RAdam 등도 있다고 하는데, 범용적으로 Adam이 많이 쓰입니다.

 

# Dropout
![image](https://github.com/user-attachments/assets/62f4d405-4109-43b2-b940-06e5c374ed4e)
`https://www.slideshare.net/slideshow/ss-79607172/79607172`
 

그래서 옵티마이저로 잘 오차를 줄였어도 과적합을 막을 수 있냐? 는 아니죠...

 

그래서 과적합 방지용으로 Dropout이 있습니다.\
일부러 학습시킬 때 몇몇 요소들의 영향을 없애면서 학습을 시키는 것입니다.\
특정 노드를 꺼버린다고 표현하는데, 특정 노드에서의 가중치를 0으로 만들어서 효과를 내지 못하게 만듭니다.

 

학습 시에만 사용하고, 실제 테스트 때에는 사용하지 않습니다.

# FNN (Feed-forward Neural Network)

FNN은 사실상 예전에 공부를 했던 다중 퍼셉트론과 일맥상통합니다.

```
from keras.layers import Conv2D, Input, MaxPool2D, Flatten, Dense, Activation, BatchNormalization
from keras.regularizers import l2
from keras.models import Sequential

model = Sequential()

img_shape = (28,28,1)

model.add(Input(shape=img_shape))

model.add(Conv2D(filters=6, kernel_size=3, activation='relu'))
model.add(MaxPool2D(2))

# 배치 정규화 포함
model.add(BatchNormalization())

model.add(Conv2D(filters=12, kernel_size=3, activation='relu'))
model.add(MaxPool2D(2))

model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(10, bias_regularizer=l2(0.01)))

model.add(Activation('softmax'))
```

이 친구처럼 여러 층을 거쳐 결과를 내는 모델을 FNN이라고 부를 수 있습니다. (CNN망을 거치긴 하지만)
```
model.summary

Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 26, 26, 6)      │            60 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 13, 13, 6)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (None, 13, 13, 6)      │            24 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 11, 11, 12)     │           660 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 5, 5, 12)       │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (None, 5, 5, 12)       │            48 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 300)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 10)             │         3,010 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ activation (Activation)         │ (None, 10)             │             0 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 3,802 (14.85 KB)
 Trainable params: 3,766 (14.71 KB)
 Non-trainable params: 36 (144.00 B)
```

이 모델 summary를 보면 이미지 하나에서 나온 파라미터 60개가 3000개까지 불어난 것을 볼 수 있습니다.\
이런 식으로 여러 망을 거쳐 더 계산을 고도화시키고 답을 도출하는 것을 인공 신경망이라고 합니다.
