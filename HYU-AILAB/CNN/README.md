# Convolutional Neural Network

## CNN 구조

* feature learning
  * convolution + relu > pooling > convolution + relu > pooling ...
* classification
  * flatten > fully connected > softmax

## Convolution

* convolution(합성곱)은 신호처리 연구에서 많이 사용되는 연산
* Input Image를 filter로 feature map을 추출
  * filter는 커널, 마스크, feature detecor 등으로도 불림
* filter 사이즈가 크다 == 수용 필드가 크다
* convolution 연산: 필터 매트릭스로 인풋 이미지를 훑으면서 곱한 값을 더해주는 것
* Stride: Convolution을 진행할 때, 필터의 이동 간격. 일반적으로 1
* Padding: Convolution으로 인한 image 모서리 부분 정보 손실 방지
  * 매트릭스가 작아지는 것을 방지
  * Zero Padding: 일반적으로 쓰이는 Padding, 모서리를 0으로 채운다
* Pooling (Subsampling)
  * 출력의 해상도를 낮춰 변형이나 이동에 대한 민감도 감소
  * 학습 노드 수가 줄어들어 학습속도 높임
  * CNN에서 일반적으로 Max Pooling 사용
* Flatten
  * feature learning 단계의 3D Vector을 1D Array로 변환
* Fully connected Layer
  * MLP 연산 실행
  * Flatten - Hidden layer1 - Hidden layer2 - Output layer
* Softmax
  * Fully connected Layer 결과값을 확률로 변환

## CNN Feedforward review

1. Input
  * 이미지를 2D 매트릭스화
  * cf) 10 * 10 이미지
2. Convolution
  * 필터로 feature map 추출
  * cf) 3 * 3 필터 3개 / Padding 있음 -> 10 * 10 * 3 3D 생성
3. Lelu
  * cf) 3D 매트릭스에서 음수 제거
4. Pooling
  * cf) 2 * 2 필터로 Maxpooling -> 5 * 5 * 3 3D 생성
5. 2~4 반복 후 flatten
  * cf) 4에서 바로 시행시 75크기의 Array 변환
6. MLP 진행
7. Softmax로 classification 완료


## Filter의 역할

  * 필터가 크면 수용 필드가 크다고 함
  * 보통 필터안의 수의 합은 같음
  * 숫자가 행별로 같으면 가로성분 추출하는 필터
  * 숫자가 열별로 같으면 세로성분 추출하는 필터
  * 가운데 부분이 크고 주변이 작으면 경계성분 추출하는 필터 

## AlexNet (2012)

2 GPU
LRN 필요

## GoogleNet (2014)

deeeep
inception module (1 * 1 conv)
auxiliary classifier

## VGGNet (2014)

Relatively simple
3 * 3 conv가 최고
19까지가 적당

## ResNet

deeeeep
vanishing & exploding gradient
resdual learning


