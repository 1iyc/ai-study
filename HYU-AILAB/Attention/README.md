# Attention

## 의미

recurrent model의 문제점을 해결하기 위해 등장

recurrent model의 문제점
  * 두 정보 사이 거리가 멀 때 해당 정보를 이용하지 못하는 문제(Long-term dependency problem)
  * 병렬 처리가 불가하여 속도가 느림(Parallelization)

## 원리

Decoder에서 출력 단어를 예측하는 매 시점 마다 Encoder에서의 전체 입력 문장을 다시 참고

동일한 비율로 참고하는 것이 아닌 해당 시점에서 예측해야 할 단어와 연관 있는 입력 단어 부분에 좀 더 집중(Attention)

cf) Ich mochte ein bier(독일어) -> I’d like a beer(영어)

  * Encoder가 ‘bier’를 받아서 벡터로 만든 결과(Encoder 출력)은 Decoder가 ‘beer’를 예측할 때 쓰는 벡터(Decoder 입력)와 유사 할 것

## Attention Function

Attention(Q, K, V) = Attention Value
  * Query에 대해 모든 Key와의 유사도를 각각 구함
  * 이 유사도를 키와 매핑 되어 있는 각각의 Value에 반영
  * 유사도가 반영된 Value를 모두 더해서 리턴 (이 값이 Attention Value)

## Dot-Product Attention

1. Attention 메커니즘 사용하여 LSTM 출력단어 예측
2. 출력단어를 예측할 때 얼마나 도움이되는지 정도를 측정
3. 2.의 수치를 Decoder로 전

## Improve

* global attention: 인코더 전체 Hidden stage에 대해 attention 측정
* local attention: 윈도우를 이용하여 대략적으로 입력문장의  단어를 추려내어 측정
* soft attention
* hard attention
