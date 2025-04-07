# 목차
- [1. 스터디 시간 필기](#스터디-시간-필기)
- [2. K-NN 총정리](#knn-총정리)
  - [2-1. 코드 분석](#코드)

# 스터디 시간 필기

 인공지능 - AI

 머신러닝 - 패턴 학습 통해서 분류 / 회귀

 딥러닝 - Neural Network 사용해 더 인간같이 추론


---

지도학습 - x 데이터 y 레이블 식으로 학습하기

ex. 강아지 / 물고기 사진

비지도학습 - 정답 없이 그냥 하는 거

ex. 클러스터링

---
Nearest Neighbor 분류

L1 거리 abs(a-b)
L2 거리 (a^2+b^2)^.5

가장 가까운 거리의 값을 return

---

KNN

K-Nearest Neighbor

knn은 k값에 따라 똑같은 학습 데이터와 데이터셋에서도 100% 정확도가 안 나올 수도..

근처에 있는 가장 자주 발견되는 데이터로 자신을 분류

K 몇으로 하는지가 좀 중요하다

성능 BAD

---

선형 분류

이미지가 있으면 (ex. 32\*32\*3)
np.reshape(-1, 1) 이런식으로 다 풀어서 학습

색깔에 치중되어 있는 방법이라 이것도 약간 별로

# KNN 총정리

![image](https://github.com/user-attachments/assets/d16b2cdf-7a42-4e44-a5d5-c75dd4ba1408)\
https://miro.medium.com/max/405/0*QyWp7J6eSz0tayc0.png

이 알고리즘은 분류 또는 회귀에 사용되는 알고리즘입니다.

 

처음 봤을 때는 FNN, CNN 같은 신경망 알고리즘인 줄 알았는데\
이름만 NN이고요, 서로 완전 별개의 것으로 보입니다.

 

K Nearest라는 이름에 걸맞게 어떤 값에 가장 근접한 k개의 이웃의 값에 따라 분류하는 방식입니다.\
k의 값으로는 1부터 데이터셋의 개수 전부 다 사용하여도 괜찮고,\
다만 동점의 상황을 대비하여 홀수로 쓰는 것이 좋습니다.

 

이웃의 거리를 비교하기 위하여 거리는\
L1 거리 (맨해튼 거리) 또는 L2 거리(유클리드 거리) 등을 사용합니다.

 

맨해튼 거리 :

(p1,p2)과 (q1,q2) 사이이면 |p1−q1|+|p2−q2|

 

유클리드 거리 : 

![image](https://github.com/user-attachments/assets/96854570-9ecf-4b69-a7fc-9b507300363e)\
https://blog.naver.com/bsw2428/221388885007
 

분류할 대상에 따라 해밍 거리 등을 사용하기도 한다고 합니다.

---

K-NN은 k=1일 때 비교군이 없기 때문에 이 알고리즘을 쓰는 의미가 거의 없고\
데이터에 따라 k 값을 조정해주면서 예측을 합니다.

 

흥미로운 특성이 하나 있는데,\
학습 데이터에 그대로 예측을 해보아도 k>1이면 정확도가 100%가 나오지 않을 수도 있습니다.\
예를 들어 우리가 찾고 싶은 데이터 근처에 노이즈가 조금 있으면 예상과 다르게 나올 수도 있겠죠.

 

그리고 K-NN은 모델을 학습할 때 파라미터를 조정하거나 내부 구조를 만들지 않습니다.\
대신, 훈련 데이터 전체를 **그대로 저장**해 두고 나중에 예측할 때 사용하는 방식입니다.

---

## 코드
코드를 보면서 어떻게 작동하는지 살펴보겠습니다.

```
class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        dists = self.compute_distances(X)
        return self.predict_labels(dists, k=k)

    # 이미지 간 거리를 구하는 코드
    def compute_distances(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]

        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i, j] = np.sqrt(np.sum(np.square(self.X_train[j]- X[i])))

        return dists

    # 가장 가까운 k개의 label을 선택, 투표
    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            closest_y = self.y_train[dists[i].argsort()[:k]]
            y_pred[i]= np.argmax(np.bincount(closest_y))
        return y_pred
```

1. **train()** 을 보면 알 수 있듯, 훈련에 아무런 의미가 없습니다.

 

 그냥 훈련 데이터를 저장해둡니다.

 

 저장된 값을 테스트 데이터와 비교하면서 레이블을 구분짓는 이 방법을 Lazy Learning이라고 부릅니다.

 

2. **compute_distances(X)**



 $$
\sqrt{\sum{(x_{\text{train}} - x_{\text{test}})^2}}
$$

 테스트 값 X를 받으면 이렇게 모든 훈련 값과 비교하는 과정을 거칩니다.\
 즉 총 train*test 회만큼 연산을 해야 하기 때문에 값이 조금만 커져도 오랜 시간이 걸립니다.\
 연산도 이미지를 예시로 들면 픽셀 개수만큼 해야 하기 때문에 엄청나게 많은 연산을 필요로 합니다.

 

3. **predict_labels(dist, k)**

 dist(compute_distances의 반환값)에서 가장 가까운 거 k개를 빼와서\
 그것들 중 가장 빈도가 높았던 것을 예측값으로 전달합니다.

---

이상입니다.

참고한 글

https://softwareeng.tistory.com/entry/cs231n-2%EA%B0%95-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EB%B6%84%EB%A5%98-34-K-%EC%B5%9C%EA%B7%BC%EC%A0%91-%EC%9D%B4%EC%9B%83-K-Nearest-Neighbors

https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
