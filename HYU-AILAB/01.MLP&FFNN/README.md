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

* 갱신 대상 학습 파라미터 <- 갱신 대상 학습 파라미터 - (learning rate * loss에 대한 파라미터의 기울기)
* Loss에 대한 𝜃의 gradient 반대방향으로 learning rate 만큼 𝜃를 업데이트 시켜 loss를 최소화하는 𝜃를 찾는다

### Stochastic Gradient Descent (SGD)

* Batch Gradient Descent: Loss function 계산시 전체 train set을 사용
- Input data가 많아 모든 training data에 대해 gradient 계산&평균 낸 후 update
- 많은 계산 량이 필요하다 -> SGD 사용
* Loss function을 계산 시 일부 조그마한 데이터의 모음에 대해서만 계산
- Batch Gradient Desent 보다 부정확하지만 빠름
- 여러번 반복 시 BGD 결과와 유사한 결과로 수렴
- Local minima에 빠지지 않고 더 좋은 결과 낼 수 있음
- 다른 알고리즘에 비해 성능 낮음

### Momentum

* Gradient Descent를 통해 이동 과정에 관성 부여
* 과거에 이동했던 방식 기억
* 그 방향으로 일정 정도 추가 이동
* Local Minima 빠져나올 수 있음

### Nesterov Accelerated Gradient (NAG)

* Momentum 방식 기초지만 Gradient 계산 방식이 다름
* momentum step을 먼저 이동했다고 가정 후 그자리에서의 gradient를 구해서 gradient step 이동
* Momentum 방식보다 효과적으로 이동가능
- 관성에 의해 훨씬 멀리 갈 수 있는 Momentum 방식의 단점 보완
- 멈춰야 할 적절한 시점에서 제동

### Adagrad (Adaptive Gradient)

* 각각의 변수마다 step size를 다르게 설정해서 이동하는 방식
* 지금까지 많이 변화하지 않았던 변수들은 step size 크게 많이 변화 한 변수들은 step size 작게
* 자주 등장하거나 변화 많이한 변수들을 optimum에 가까이 있다고 판단. 작은 크기로 이동하면서 세밀하게 값 조정
* word2vec이나 GloVe 같이 word representation 학습에 좋은 성능 거둘 수 있음
* 장점: step size decay 등을 신경 안써도 된다. step size를 0.01 정도로 사용 후 바뀌지 않음
* 단점: step size가 너무 작아져서 거의 움직이지 않게 된다. -> RMSProp, AdaDelta로 해결

### RMSProp

* Gt 부분을 합이 아니라 지수 평균으로 대체하여 Adagrad의 Gt가 무한정 커져서 움직이지 않는 현상 고침
* Gt가 무한정 커지진 않으면서 최근 변화량의 변수간 상대적인 크기 차이는 유지

### AdaDelta

* RMSProp와 동일하게 G를 구함
* step size의 변화값의 제곱을 가지고 지수평균 값을 사용 (위 두개는 learning rate가 step size)

### Adam

* RMSProp와 Momentum 방식을 합친 것 같은 알고리즘
* Momentum(지금까지 계산한 기울기의 지수평균) + RMSProp(기울기의 제곱값의 지수평균)

## Readings

Optimizer: http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html
Word Vectors from PMI Matrix: https://www.kaggle.com/gabrielaltay/word-vectors-from-pmi-matrix
