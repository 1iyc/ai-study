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



## 
