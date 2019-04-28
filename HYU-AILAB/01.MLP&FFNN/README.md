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


