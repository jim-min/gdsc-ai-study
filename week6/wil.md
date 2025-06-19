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



## kernel_size


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



