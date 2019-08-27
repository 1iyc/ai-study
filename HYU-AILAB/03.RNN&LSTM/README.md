# RNN & LSTM

## Recurrent Neural Networks (순환신경망)

* 히든 노드가 directed cycle을 형성하는 인공신경망의 한 종류
* 음성, 문자 등 순차적으로 등장하는 데이터 처리에 적합

## RNN 기본 동작

* yt = Why * ht + by
  * yt는 t시점에서 인풋인 xt에 대한 결과
* ht = tanh(Whh * ht-1 + Wxh * xt + bh)
  * ht는 t시점의 히든 레이어
  * ht-1은 t시점 히든 레이어의 전 레이어
* xt
  * 인풋값

## RNN backward pass

* Gradient swapper
  * f(x, y) = xy -> df / dx = y ; df / dy = x
* Gradient distributor
  * f(x, y) = x + y -> df / dx = 1 ; df / dy = 1
* 하이퍼볼릭탄젠트 함수 미분: tanh(x) -> 1 - tanh^2(x)

## Long Short-Term Memory models (LSTM)

* 관련 정보와 그 정보를 사용하는 지점 사이 거리가 멀 경우 학습능력이 저하되는 RNN의 특성 보완
  * Vanishing gradient problem
* Hidden state와 cell state 존재

## LSTM 기본동작

* forget gate
  * 잊을지 말지 정함 (0~1)
  * sigmoid(W * x + W * ht-1 + b)
* input gate
  * 현재 정보를 기억하기 위한 게이트
  * sigmoid(W * x + W * ht-1 + b) ⊙ tanh(W * x + W * ht-1 + b)
  * sigmoid(W * x + W * ht-1 + b) 는 (0~1) 강도를 뜻함
  * tanh(W * x + W * ht-1 + b) 는 (-1~1) 방향을 뜻함
  * ⊙ 는 Hadamard 연산 아다마르 곱 / 행렬의 대응되는 각 요소를 곱함
* cell state update
  * forget gate * pre cell state + input gate
* hidden state update
  * sigmoid(W * x + W * ht-1 + b) * tanh(cell state update)

## CNN + RNN

* CNN에서 Fully Connected한 것을 RNN의 첫번째 히든 레이어로 

## GRU

* forget, input gate 하나로 통합 -> 웨이트 파라미터 수가 줄어듦

## Readings

* https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/
