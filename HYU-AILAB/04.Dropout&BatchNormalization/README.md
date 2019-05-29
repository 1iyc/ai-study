# Dropout & Batch-Normaliztaion


## Dropout

* overfitting을 해결하기 위한 작업

## Overfitting
* layer가 많아질 때 나타나는 문제점; 지나치게 분류를 잘하려고 선을 많이 구부림
* 확인법: 트레이닝 셋에는 error가 낮지만 실제 데이터에 대해 오차가 큼
* 해결방법
  * training data 많이 모으기: training set, validation set, test set으로 나누어 학습 진행 -> overfitting을 판단하기 쉬워짐
  * feature 갯수 줄이기: 딥러닝에선 중요하지 않다. 이유는 sigmoid 대신 LeRU 계열을 사용하며 dropout을 통해 feature을 스스로 줄인다
  * regularization: weight가 너무 큰 값을 갖지 않도록 제한. weight가 커지지 않아 선이 구부러지는 형태를 피할 수 있다.


## Regularization

* 단순하게 학습에서 cost function이 작아지는 쪽으로만 진행하게 되면 특정 가중치 값들이 커지면서 결과가 나빠지는 경우가 있다.
* 특정 가중치에 대한 의존을 줄이는 작업 == 파라미터들의 분산을 줄인다.
* cf) L2 Normalization


## Readings

* Dropout
  * https://pythonkim.tistory.com/42
  * https://blog.naver.com/PostView.nhn?blogId=laonple&logNo=220542170499

