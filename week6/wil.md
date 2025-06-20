# 목차

- [0. CNN](#cnn)
- [1. 파라미터 소개](#파라미터-소개)
  * [kernel_size](#kernel-size)
  * [padding](#padding)
  * [stride](#stride)
  * [dilation](#dilation)
- [2. LeNet](#lenet)
- [3. AlexNet](#alexnet)

# CNN

https://youtu.be/_d9pCrZNnYk?si=3XOSRpdPoPFe_Yha

임커밋님의 영상을 보고 기본적인 개념은 쉽게 잡을 수가 있었어요.\
아예 처음이면 위 영상을 보는 것을 추천합니다.


CNN은 입력에 대하여 특정 크기의 커널만큼의 연산을 하여 출력층을 만드는 과정입니다.\
여기서 1차원 벡터화하기 전의 출력층을 feature map이라고 부릅니다.

이러한 연산을 하는 이유는 데이터 입력에서의 주변값과의 유사성을 반영하기 위해서입니다.\
일반적인 신경망은 딱 그 벡터의 값만 계산하게 되는데, CNN은 주변값을 함께 계산하게 되기 때문에
훨씬 더 '맥락'에 대한 이해 능력이 높습니다.\
이에 따라 NLP, 이미지 처리 등에 사용되고 있습니다.



# 파라미터 소개
연산에 차이를 줄 수 있는 파라미터가 몇 개 있습니다.




`torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)`



PyTorch의 2차원 CNN 모델을 보겠습니다...



빨간색 칠한 부분을 보자면 크게\
*kernel_size, stride, padding, dilation, groups* 등이 있네요!



## kernel size


kernel_size는 기본적으로 한 연산에서 계산할 범위를 말합니다. 필터라고도 부릅니다.\
커널 사이즈가 1이면 그냥 FNN이랑 딱히 다를 게 없겠죠.



사이즈는 1d면 3, 2d면 3*3 이런 식으로 지정할 수 있습니다.

보통 3, 5와 같은 홀수(2n + 1)로 지정하는데 그 이유는 한 데이터를 기준으로 양 옆으로 n개의 데이터를 참조하기 위함입니다.\
kernel_size는 padding * 2 + 1의 크기를 가지는데 이건 아래에서 더 설명할게요.

## padding


padding은 feature map이 작아지는 것을 방지하기 위해 추가됩니다.




![image](https://github.com/user-attachments/assets/caff5c6d-06fa-4a68-b848-4cfae6ad9ecc)

*출처 : GDG Hongik week 6 강의록*


행렬의 가장 끝자락부터 계산을 시작하면, 이렇게 feature map이 5x5에서 3x3으로 줄어드는 현상을 볼 수 있습니다.\
커널이 5x5 사이즈였다면 feature map이 1x1이었겠죠.



그래서 값이 0인 padding을 씌워 feature map의 크기를 일정하게 유지합니다.\
패딩에 넣을 값은 padding_mode로 조절할 수 있어 보입니다.



padding의 크기는 kernel_size // 2 로 맞춥니다.\
아까 2*padding + 1의 반대로 생각하면 되겠네요!



## stride
stride는 커널이 이동하는 간격을 의미합니다.\
stride는 기본적으로 1이고 옆으로 한 칸씩 이동하는데 이 값을 n으로 늘려주게 되면\
출력층을 n배씩 줄일 수 있게 됩니다.



## dilation
dilation은 커널의 넓이를 늘리는 것입니다.\
stride가 연산마다의 거리를 넓혀주는 것이면,
dilation은 연산 내에서 커널이 보는 범위를 늘려주는 것입니다.



연산할 때 참고할 데이터의 파라미터는 유지하되 더 넓게, 효율적으로 볼 수 있는 방법입니다.



참고로 이 방법을 사용하면 padding도 맞춰서 키워줘야 합니다.

![image](https://github.com/user-attachments/assets/162d9e06-07c1-4585-8e85-99433a6084f9)


공식은 이러한 방식을 따릅니다.
`2 * padding = dilation * (kernel_size - 1)`
이 성립해야 shape가 유지됩니다.

이런 CNN망을 거친 후에, MaxPooling 층 같은 것을 거쳐 연산 사이즈를 줄이고,\
Flatten 후 출력층으로 나오는 것이 일반적인 CNN입니다.

# LeNet
 

유명하고 효과적인 AlexNet을 공부하기 전에 그 바탕이 되고 더 먼저 개발된 LeNet에 대해 알아보겠습니다.\
LeCun이 고안한 LeNet-5(1998)은 CNN의 존재 이유를 아주 잘 이해한 이미지 분류 모델입니다.

 

그 전까지 이미지 분석 모델은 이미지를 FC로 분해해 학습했기 때문에, 파라미터가 너무 많았고, 이미지의 주변 픽셀 간의 상관관계를 무시했기 때문에 비효율적이고 성능도 좋지 않았습니다.

![image](https://github.com/user-attachments/assets/58ec4f6d-03f5-48dc-a9ef-5119227e0e9b)


LeNet에서는 그런 단점을 극복하기 위하여 위 형태를 제시합니다.

 

- C1. 32x32 이미지를 28x28 사이즈의 6개의 feature map으로 치환하고,
- S2. stride가 2인 연산을 통해 14x14로 줄이고,
- C3. 다시 10x10 짜리 16개의 층으로 만들고,
- S4. stride 2인 층으로 다시 5x5 사이즈로 만듭니다.
- C5. 여기서 16개의 feature map을 120개의 커널을 통해 1x1x120의 층으로 만듭니다.
- F6. 마지막으로 FC layer로 만들어주고 (tanh를 사용하여) 결과를 출력합니다.

![image](https://github.com/user-attachments/assets/3e7b7df6-9a25-4809-aa9b-5a38ba997a8e)

이런 식으로 나온다고 하네요

C3 layer 학습 방식이 특이한데, 

![image](https://github.com/user-attachments/assets/bcbac3e0-0c99-4465-bdeb-16b2af15123b)

이런 식으로 S2 feature map을 전부 사용하는 것이 아니라 특정 feature map만 빼와서 서로 간의 connection을 줄였습니다.

# AlexNet

https://youtu.be/40Gdctb55BY?si=6gBKKRUDmdrDuThy
https://www.youtube.com/watch?v=5i2xG4WqR7c

(참고한 영상)

AlexNet은 처음으로 ReLU 비선형 함수를 도입한 모델이라고 하는데요.\
AlexNet을 기준으로 이미지 분석에 엄청난 딥러닝 붐이 왔다고 해요.

 

정확도가 급격하게 올라가기 시작한 이 방법에 대해 알아보겠습니다.
 

*ReLU 사용*을 통해 6배 빠르게 오류율을 줄일 수 있었다고 합니다.

 

또한 *2대의 GPU*를 동시에 사용했는데,
메모리 사용을 늘리기 위하여 사용했다고 합니다.

 

*LRN layer*이라는 것을 사용을 했다고 합니다. 한국말로는 지역 응답 정규화라고 하는데요.
feature map 각각의 뉴런에 대해 주변 뉴런들 활성화 정도에 의해 정규화를 수행하는 것이라고 합니다.

![image](https://github.com/user-attachments/assets/02a614cc-fa34-4101-837b-58cba5aef0ee)

LRN 공식은 이렇게 됩니다.


그 외에 Pooling을 겹치게 하는 Overlapping Pooling, 데이터 증강, Dropout과 같은 요즘에 많이 쓰이는 방식을 다 도입했습니다.


위 영상에서 논문 리뷰를 열심히 해주신 게 있어서 자세한 건 더 살펴보시면 될 듯 합니다 ㅎㅎ
