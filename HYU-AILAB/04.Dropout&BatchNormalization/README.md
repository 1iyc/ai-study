# Dropout & Batch-Normaliztaion


## Dropout

* overfitting을 해결하기 위한 작업
* 전체 weight를 계산에 참여시키는 것이 아닌 layer에 포함된 weight 중 일부만 참여
* 특정 뉴런을 난수를 사용해 0으로 만듦 -> 제외한 것과 같은 효과가 낢
* weight마다 귀를 판단, 꼬리를 판단, 발톱을 판단 등의 전문가들이 있음
  * 사공이 많으면 배가 산으로 간다는 상황을 방지하기 위해 weight들을 dropout

## Dropout 효과

* Voting 효과
  * voting에 의한 평균 효과를 얻을 수 있음
  * 결과적으로 regularization과 비슷한 효과
*  Co-adaptation을 피하는 효과
  * 어떤 뉴런의 가중치나 바이어스가 특정 뉴런의 영향을 받지 않아 뉴런이 동조화되는 것을 피할 수 있음
  * robust한 망을 구성할 수 있다.

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
  * l2reg = 0.001 * tf.reduce_sum(tf.square(W)) : 람다의 값은 보통 0.001. 중요할수록 0.01 혹은 0.1
  * weight의 제곱 합계이기 때문에 0.001로도 충분한 영향력 행사 가능

## Batch Normalization 효과

* DNN의 문제중 하나인 Internal Covariate Shift 해결
  * 히든레이어가 높아질 수록 다른 히든레이어 값을 가지게 되는 경우가 높아질 것이다.
  * 웨이트의 영향을 계속 받기 때문
* BN 이전에 위의 문제 해결법
  * careful initialization - 하기 어렵다
  * small learning rate - 학습 완료가 느리다

## Batch Normalization

* activation function에 들어가는 input의 range를 제한하는 방법
* 기존 방식
  * wx + b -> Activation -> Hidden Node
* Batch Norm
  * wx + b -> BN
* 기존의 분포와 비슷한 분포를 가지게 된다
* Activation Function별 BN
  * ReLU: 0을 기준으로 대칭된 값을 많이 사용 살짝 엇나가도 됨
    * 분산이 1, 상징적으로 조그마한 베리언스
    * 분산이 크면 Internal Covariate Shift가 일어날 수 있음
    * 액티베이션을 좀 살리기 위해 오른쪽으로 scale and shift하기도 함
  * Sigmoid: 양 극단이 너무 값이 극단이라(vanishing) 0 주변값 사용
    * 분산 1 혹은 줄여주는게 좋다
    * 분산이 크면 Internal Covariate Shift, Vanishing Gradient가 있을 수 있음
    * 대표적으로 지로빈의 유닛베리언스를 쓴다 ???
    * 범위를 작게 잡으면 선형적으로 되어버린다
* 노말라이징: (xi - xi들의 mini-batch mean) / root(xi들의 mini-batch 분산 + 에타 상수값)

## Batch Normalization의 장점

* learning rate를 올려도 안정적이다 -> Cvariation Shift 문제가 해결되었기 때문
* initilalization에 크게 신경쓰지 않아도 된다
* Regularization Effect -> Dropout이 크게 필요하지 않다
  * 미니배치에 들어가는 평균과 분산, 벡터값들이 지속적으로 변하는 과정에서 분포가 살짝씩 바뀌기 때문  * 화이트노이즈 효과?

## Exponential Linear Unit (ELU)

* x (if x > 0) ; ae^x - a (if x <= 0)
* input이 음수일때 값이 -1~0
* 분포 변화없이 지속적으로 값을 내주기 때문에 batch norm이 필요 없음

## Scaled Exponential Linear Unit (SELU) ???

* lamda = x (if x > 0) ; ae^x - a (if x <= 0)  ; alpha: 1.6733, lambda: 1.0507
* batch norm이 필요 없음
* 스케일하는 감마, 알파값을 조금 조정하면 평균 0 분산 1로 수렴한다는 것을 밝힘
* 기존에 다른 batch norm이나 방법들보다 훨씬 좋은 성능 나타냄

## Readings

* Dropout
  * https://pythonkim.tistory.com/42
  * https://blog.naver.com/PostView.nhn?blogId=laonple&logNo=220542170499

## Batch Normalization

* https://www.youtube.com/watch?v=TDx8iZHwFtM
