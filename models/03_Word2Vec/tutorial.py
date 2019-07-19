import tensorflow as tf
import sys
import os
import argparse
from six.moves import xrange
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def word2vec_basic(log_dir):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    font_name = matplotlib.font_manager.FontProperties(
        fname="/usr/share/fonts/NanumFont/NanumGothic.ttf"
    ).get_name()
    matplotlib.rc('font', family=font_name)

    sentences = ["나 고양이 좋다",
                 "나 강아지 좋다",
                 "나 동물 좋다",
                 "강아지 고양이 동물",
                 "여자친구 고양이 강아지 좋다",
                 "고양이 생선 우유 좋다",
                 "강아지 생선 싫다 우유 좋다",
                 "강아지 고양이 눈 좋다",
                 "나 여자친구 좋다",
                 "여자친구 나 싫다",
                 "여자친구 나 영화 책 음악 좋다",
                 "나 게임 만화 애니 좋다",
                 "고양이 강아지 싫다",
                 "강아지 고양이 좋다"]

    def sentences2voca(sentences):
        word_sequence = " ".join(sentences).split()
        word_list = " ".join(sentences).split()
        word_list = list(set(word_list))
        word_dict = {w: i for i, w in enumerate(word_list)}

        return word_sequence, word_dict

    word_sequence, word_dict = sentences2voca(sentences)

    def make_skip_grams(word_sequence, word_dict):
        skip_grams = []

        for i in range(1, len(word_sequence) - 1):
            target = word_dict[word_sequence[i]]
            context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]

            for w in context:
                skip_grams.append([target, w])

        return skip_grams

    skip_grams = make_skip_grams(word_sequence, word_dict)

    def random_batch(data, size):
        random_inputs = []
        random_labels = []
        random_index = np.random.choice(range(len(data)), size, replace=False)

        for i in random_index:
            random_inputs.append(data[i][0])  # target
            random_labels.append([data[i][1]])  # context word

        return random_inputs, random_labels

    #########
    # 옵션 설정
    ######
    # 학습을 반복할 횟수
    training_epoch = 300
    # 학습률
    learning_rate = 3e-5
    # 한 번에 학습할 데이터의 크기
    batch_size = 20
    # 단어 벡터를 구성할 임베딩 차원의 크기
    # 이 예제에서는 x, y 그래프로 표현하기 쉽게 2 개의 값만 출력하도록 합니다.
    embedding_size = 2
    # word2vec 모델을 학습시키기 위한 nce_loss 함수에서 사용하기 위한 샘플링 크기
    # batch_size 보다 작아야 합니다.
    num_sampled = 15
    # 총 단어 갯수
    voc_size = len(word_dict)

    graph = tf.Graph()

    with graph.as_default():

        with tf.name_scope("inputs"):
            train_inputs = tf.placeholder(tf.int32, name="word_batch", shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, name="labels", shape=[batch_size, 1])

        with tf.name_scope("embeddings"):
            embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
            selected_embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        with tf.name_scope("weights"):
            nce_weights = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))

            tf.summary.histogram("nce_weights", nce_weights)

        with tf.name_scope("biases"):
            nce_biases = tf.Variable(tf.zeros([voc_size]))

            tf.summary.histogram("nce_biases", nce_biases)

        with tf.name_scope("loss"):
            loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                               biases=nce_biases,
                               labels=train_labels,
                               inputs=selected_embed,
                               num_sampled=num_sampled,
                               num_classes=voc_size))

            tf.summary.scalar("loss", loss)

        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        merged = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

    num_steps = 300001

    with tf.compat.v1.Session(graph=graph) as session:
        writer = tf.summary.FileWriter(log_dir, session.graph)

        init.run()
        print("Initialized")

        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = random_batch(skip_grams, batch_size)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            run_metadata = tf.RunMetadata()

            _, summary, loss_val = session.run([optimizer, merged, loss],
                                               feed_dict=feed_dict,
                                               run_metadata=run_metadata)
            average_loss += loss_val

            writer.add_summary(summary, step)

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000

                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

            if step == (num_steps - 1):
                writer.add_run_metadata(run_metadata, 'step%d' % step)

        saver.save(session, os.path.join(log_dir, 'model.ckpt'))

        trained_embeddings = embeddings.eval()

    writer.close()

    for i, label in enumerate(word_dict):
        x, y = trained_embeddings[i]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2),
                     textcoords='offset points', ha='right', va='bottom')

    plt.show()


def main(_):
    current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(current_path, 'log'),
        help='The log directory for TensorBoard summaries.')
    flags, unused_flags = parser.parse_known_args()

    word2vec_basic(flags.log_dir)

    return 1


if __name__ == '__main__':
    tf.app.run()
