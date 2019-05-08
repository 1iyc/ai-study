# 01. MLP & FFNN

Multi Layer Perceptron & Feed Forwad Neural Network

## 단일 퍼셉트론

* 뇌의 뉴런을 모방
* 인풋과 가중치들의 곱(w0x0, w1x1, w2x2)들의 합에 Bias 값을 더한다
* 위의 값을 Activation Function(활성 함수)에 넣어서 결과 출력

## Activation function

* 뉴런에서 계산된 값이 임계치보다 크면 1 출력 작으면 0 출력
* 다음으로 신호를 보내는지 결정

## 가중치와 Bias 조정

* ∣목표값 - 출력값 ∣* 인풋값 * learning rate + W(t) = W(t+1)
* Bias는 W(t)를 B(t)로 변경
* 변화가 없거나 미미할 정도 까지 인풋을 계속 학습

## Tutorial and.py

## 다층 퍼셉트론

* 입력층과 출력층 사이에 1개 이상의 hidden layer 추가
* 대부분 역전파 알고리즘으로 학습

## 역전파

* loss를 줄이는 것이 목표
* w = w - learning rate * (𝜕E / 𝜕w)

## MLP의 한계

* 항상 global minimum을 찾는다는 보장이 없다.
* 가중치들의 초기설정이 성능에 큰 영향을 미친다.
* Layer의 개수와 node의 개수를 정하는 모델이 없다.
* Layer가 많다고 성능이 좋아지는 것이 아니다.

## Activation function

* 네트워크에 비선형을 추가하기위해 사용됨
* H(X) = f(g(i(j(X)))) -> H(X) = WX 결국 층이 많아도 선형적이게 된다
* Activation function 추가시 각 층의 출력 결과를 비선형화 하고 다음 층에 넘겨준다.
- H(X) = af(f(af(g(af(i(af(j(X))))))))
+ af는 activation function

## Activation function 종류

* step function
* sigmoid
* hyperbolic tangent
* ReLU
* softmax

### step function

* 임계값 기준으로 활성화 혹은 비활성화되는 형태
* step(x) = 1 (x >=0)
* step(x) = 0 (otherwise)

### sigmoid

* sigmoid(x) = 1/(1+e^-x)
* 미분이 쉬움
* 결과 값이 [0, 1]로 제한
* 입력이 작을 때 0, 클 때 1에 수렴
* 역전파 신경망에 많이 쓰임
* 가중치나 바이어스를 조금 변화 시켰을 때 출력이 조금씩 변화하도록 만들 수 있다.
* Gradient Vanishing
- 그라이언트가 죽어서 학습이 되지 않는다.
- 이전 레이어로 전파되는 그라디언트가 0에 가까워지는 현상
- 양 극단의 미분값이 0에 가깝기 때문에 발생한다.
- 레이어를 깊게 쌓으면 파라미터의 업데이트가 제대로 이루어지지 않는다
- ReLU나 초기화를 잘 하여 극복 가능
* 지수함수라서 계산이 복잡하다.

### LeLU

* LeLU(x) = x (x >= 0)
* LeLU(x) = 0 (otherwise)
* 0에서 확 꺾이기 때문에 비선형
* 계산이 효율적
* Sigmoid보다 계산속도가 빠름
* 음수가 나온 노드는 학습이 불가능
- 간단한 네트워크에서는 성능저하
- leaky ReLU의 존재
* 0에서 미분 불가능
- softplus함수로 해결: 0근처에 떨어지는 데이터들 많을 경우 성능 증가

### Leaky ReLU

* Leaky ReLU(x) = x (x >= 0)
* Leaky ReLU(x) = 0.01x (otherwise)
* ReLU 함수의 변형, 음수에 대해 1/10로 값을 줄여 사용

### softmax

* 입력된 값들을 0~1 사이의 정규화된 값으로 출력
* 출력 값들의 총합은 항상 1
* 주로 출력 노드에서 사용
* 분류, 강화학습에 사용
* vs sigmoid
- sigmoid는 해당 뉴런으로 들어오는 입력들과 bias에 의해 출력 결정
- softmax는 다른 뉴런의 출력 값들 과의 상대적인 비교를 통해 출력이 결정

## Loss Function

### Mean Squared Error (평균 제곱 오차)

* 회귀의 손실 함수
* 모든 출력층 뉴런의 값이 계산에 들어간다.

### Cross Entropy Error (교차 엔트로피 오차)

* 분류에서 소프트맥스 함수의 손실함수로 쓰임
* 정답에 해당하는 위치의 뉴런 값만 계산에 들어간다.

## Optimizer

### Gradient Descent

* 갱신 대상 학습 파라미터 <- 갱신 대상 학습 파라미터 - learning rate * loss에 대한 파라미터의 기울기

