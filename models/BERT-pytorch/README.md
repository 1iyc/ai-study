# pytorch-BERT 1.1.0

git clone from https://github.com/huggingface/pytorch-transformers

# train-v1.1.json

## 예시

```json
{
  'paragraphs': [
    {
      'qas': [
          {
            'answers': [
              {
                'answer_start': 0,
                'text': 'Answer1'
              }
            ],
            'question': 'What is Answer1?',
            'id': '1a2s3d6f...'
          },
          {
            'answers': [
              {
                'answer_start': 20,
                'text': 'Answer2'
              }
            ],
            'question': 'What is Answer12',
            'id': '4a2b3c6d...'
          },
          ...
        ],
      'context': 'context1'
    },
    {
      'qas': [],
      'context': 'context2'
    },
    ...
  ],
  'title': 'Title'
}
```

## 특징

* 한 paragraphs는 여러 개의 context를 가지고 있음
* 한 context는 여러개의 Q&A를 가지고 있음
* 한 paragraphs 당 공통의 주제 혹 제목[title]을 가지고 있으나 모든 context가 전부 같은 곳에서 발췌된 것 같지 않음
* 한 qas는 한개 이상의 답(정답 시작 Index[answer_start] 포함)과 질문, id를 포함
* 한 answers는 한 질문에 대해 여러개의 답을 가질 수 있을 것 같지만 1.1에서는 보이지 않음. 2.0에서 특정 단어를 포함하지 말아야하거나 다른 문장에서 합쳐야하는 경우 쓸 수도 있을 것 같음

# utils_squad.read_squad_examples

example = SquadExample(
  qas_id=qas_id,
  question_text=question_text,
  doc_tokens=doc_tokens,
  orig_answer_text=orig_answer_text,
  start_position=start_position,
  end_position=end_position,
  is_impossible=is_impossible)
