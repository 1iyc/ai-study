# SK ai.x 2019

## Smart Interfaces for Human-Centered AI / James Landay (Stanford University)

AI technology should be ispired by human instelligence.

The development of AI should be guided by its human impact.

The goal of AI should be enhancing ...

* AI Needs user experience (ux) design

How do we design the UX?

Amazon Echo, MS cortana & other Smart Speaker use Voice UI

How do we design them to deal with natural human conversation?

How do we design to support multimodal input?

Computer vision-based skin cancer detection getting better and better.

* How do we find balance between instructing systems on every small actions?

* How do we invent the important interface metaphors that will make understanding and using these systems easier?

  * WIMP, GUI

## Deep Generative Models for Speech / Heiga Zen (Google Brain)

* WaveNet: A generative model for raw audio

Autoregressive modelling of speech signals

WaveNet

modeled by CNN

Key components

Causal dilated convolution

Gated convolution residual skip

Softmax at output

* Causal dilated convolution

Too long to be captured by normal RNN/LSTM

Dilated convolution

* Non-linearity

* Softmax

sampling & quantizing

* WaveNet for modeling speech signals

Advantage

* Autoregressive

Disadvantage

* Sequence of small computations

* sample-by-sample slow

* Inverse autoregressive flow (IAF)

Gaussian autoregressive function as invertible transform for flow

* Probability density distillation

input -> student(IAF Parallel WaveNet) -> output(input to teacher) -> teacher(AR WaveNet)

* Tacotron

* WaveNet-based low bit-rate speech coding

* Parrotron - Normalization of hearing-impaied speech

Convert atypical speech from a deaf speaker to fluent speech

<https://google.github.io/tacotron/publications/>

* Trnaslatotron - Spanish-English speech translation

## AI for Visual Pleasure / Tae Hyun Kim

* Blu-ray (1920x1080) vs 88'', 8K (7680x4320)

  * upscale

* Computational Imageing

  * High-quality image generation

    * Capture Protocol + Sensor + ->

  * Indirect image formation from measurements using algorithms

    * 노이즈없애기
    * 촬영한거 이쁘게
    * 어두운데서 촬영한 것 잘보이게
    * 인물사진 (Manipulated Deep)

* Wha we do: Deblurring
  * Remove motion blur
    * 손떨림 등
  * Low-resolution image -> High-resolution

* How can we employ AI techniques for deblurring?
  * The essential AI handbook for leaders (Book)
    * To move

* Step1: Acquire input-output pair
  * Large-scale dataset acquisition
    * Generate realistic blurs by averaging sharp videos with a high-speed-camera
* Step 2: Build a Network
  * Baseline
    * Very deep CNN model
* Step 3: Customization
  * Deep multi-scale CNN
    * To handle large motion (100x100 pixels), need a network of ~50 layers with 3x3 filters
  * Multi-scale approach
* Step 4: Training
  * Multi-scale content loss
  * ...

* Video

* Step1
* Step2-3: Build a Network & Customization

* Generalization: Test with real scene
  * Input: Low quality real YouTube video -> Output: High quality deblurred

* Ablation Study
  * SISR+STTN
  * STTN+VDSR(CVPR16)

* Recent SR Methodologies
  * Table from: Hui et al. CVPR18
  * IDN / MemNet / DRRN / LapSRN / DRCN / VDSR / Bicubic

* Our Observation
  * At least 0.35dB margin with the state-of-the-art method for 2x SR task!

## Frontiers of Metric-based Few-shot Learning / Jake Snell (Univ. of Toronto)

* Matching Networks
  * Computes a normalized similarity of the query to each support example

* Limitations of Metric-based FSC
  * Less adaptaion
  * ...

* Generalized Few-shot Classification
  * 
* Bridging the Gap from Few-shot to Many-shot

## Meta AI Research and System Development / Dongyeon Cho (SK telecom)

### Meta AI Project in T-brain

* Applications / Scalable Cluster infrastructure / Hyperparameter Optimization / Neural Architecture Search / Learning Algorithms

### Gradient-Based Meta-Learning

* Learning good initialization
  * MAML / Reptile

* Neural Architecture Search (NAS)
  * Progressive NAS
    * 좋은 셀이라고 예상되는 셀에서만 evaluation 수행

* Experimental Results

### Our Method

* Building a base network

* Learning disparate modulators

* Building a selection network OR ensemble

* Supervision for training a selection network

## Towards Scalable Conversational AI system & Taeyoon Kim (SK telecom)

* 봇이 세상과 연결고리가 없다

### BERT

* 트랜스포머라는 RNN을 대체 작은 신경망 블록을 아주 깊이 쌓아올림
* Encoder만 사용 Attention Model에서
* 모델의 크기가 엄청나게 크다 파라미터 1억개 이상

#### KoBERT: Tokenization

* SentencPiece Tokenization

### SUMBT: Slot-Utterance
