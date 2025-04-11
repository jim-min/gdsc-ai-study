# 스터디 필기

## 이미지의 채널값

- RGB\
  R, G, B 각각 0 ~ 2^8 - 1의 값을 가짐
  
- HSV\
  hue, saturation, value\
  0\~360, 0\~100, 0\~100

## 이미지 처리

- Detection
- Segmentation
- Image captioning

### 신경써야 할 것
  시점, 밝기 변화, 변형됨, 사진의 일부분, 배경, 디테일한 분류

## Loss Function

### 경사하강법

Loss 함수에 대해 변화율 계산해가면서 loss 낮추려고 적응시킴

- Hinge Loss 함수
- Softmax 함수 & Cross-entropy loss 함수

## 정규화

- Overfitting (과적합)
  데이터 변하면 의미 없어짐

### 어떻게 피해야 하나

Epoch 적당히 조절하면서 정규화

- L1 정규화\
  특정 필드 0으로 만들어서 반영 안 되게 만듦

- L2 정규화\
  좀 더 튀는 값에 대해 패널티를 더 줌
