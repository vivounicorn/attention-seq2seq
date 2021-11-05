import os

from sklearn.model_selection import train_test_split
import time
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import utils.preprocessor as up
from Decoder import *
from Encoder import *
import pickle


class DatasetParams:
    def __init__(self):
        self.inp_lang = None
        self.targ_lang = None
        self.max_length_targ = 0
        self.max_length_inp = 0
        self.BATCH_SIZE = 64
        self.BUFFER_SIZE = 0
        self.steps_per_epoch = 0
        self.embedding_dim = 256
        self.units = 1024
        self.vocab_inp_size = 0
        self.vocab_tar_size = 0


class ModelParams:
    def __init__(self):
        self.encoder = None
        self.decoder = None
        self.optimizer = None
        self.optimizer = None
        self.loss_object = None


class CheckPointsParams:
    def __init__(self):
        self.checkpoint_prefix = ''
        self.checkpoint = None
        self.checkpoint_dir = ''


class Modelling:
    def __init__(self):
        pass


def data_preparing(path_to_file):
    params = DatasetParams()

    # 尝试实验不同大小的数据集
    num_examples = 30000
    input_tensor, target_tensor, params.inp_lang, params.targ_lang = up.load_dataset(path_to_file, num_examples)
    # 计算目标张量的最大长度 （max_length）
    params.max_length_targ, params.max_length_inp = up.max_length(target_tensor), up.max_length(input_tensor)
    # 采用 80 - 20 的比例切分训练集和验证集
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                    target_tensor,
                                                                                                    test_size=0.2)
    # 显示长度
    print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))
    print("Input Language; index to word mapping")
    up.convert(params.inp_lang, input_tensor_train[0])
    print()
    print("Target Language; index to word mapping")
    up.convert(params.targ_lang, target_tensor_train[0])
    params.BUFFER_SIZE = len(input_tensor_train)

    params.steps_per_epoch = len(input_tensor_train) // params.BATCH_SIZE

    params.vocab_inp_size = len(params.inp_lang.word_index) + 1
    params.vocab_tar_size = len(params.targ_lang.word_index) + 1
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(params.BUFFER_SIZE)
    dataset = dataset.batch(params.BATCH_SIZE, drop_remainder=True)

    return params, dataset


def inference_model_building(dataset_params):

    params = ModelParams()

    params.encoder = Encoder(dataset_params.vocab_inp_size, dataset_params.embedding_dim, dataset_params.units,
                             dataset_params.BATCH_SIZE)

    params.decoder = Decoder(dataset_params.vocab_tar_size, dataset_params.embedding_dim, dataset_params.units,
                             dataset_params.BATCH_SIZE)

    params.optimizer = tf.keras.optimizers.Adam()
    params.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    return params


def model_building(dataset_params, dataset):
    example_input_batch, example_target_batch = next(iter(dataset))
    print(example_input_batch.shape, example_target_batch.shape)

    params = ModelParams()

    params.encoder = Encoder(dataset_params.vocab_inp_size, dataset_params.embedding_dim, dataset_params.units,
                             dataset_params.BATCH_SIZE)
    # 样本输入
    sample_hidden = params.encoder.initialize_hidden_state()
    sample_output, sample_hidden = params.encoder.call(example_input_batch, sample_hidden)
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))
    attention_layer = BahdanauAttention(10)
    attention_result, attention_weights = attention_layer.call(sample_hidden, sample_output)
    print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))
    params.decoder = Decoder(dataset_params.vocab_tar_size, dataset_params.embedding_dim, dataset_params.units,
                             dataset_params.BATCH_SIZE)
    sample_decoder_output, _, _ = params.decoder.call(tf.random.uniform((64, 1)),
                                                      sample_hidden, sample_output)
    print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
    params.optimizer = tf.keras.optimizers.Adam()
    params.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    return params


def loss_function(model_params, real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = model_params.loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def chk_settings(model_params):
    params = CheckPointsParams()
    params.checkpoint_dir = './training_checkpoints'
    params.checkpoint_prefix = os.path.join(params.checkpoint_dir, "ckpt")
    params.checkpoint = tf.train.Checkpoint(optimizer=model_params.optimizer,
                                            encoder=model_params.encoder,
                                            decoder=model_params.decoder)
    return params


@tf.function
def train_step(dataset_params, model_params, inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = model_params.encoder.call(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([dataset_params.targ_lang.word_index['<start>']] * dataset_params.BATCH_SIZE, 1)

        # 教师强制 - 将目标词作为下一个输入
        for t in range(1, targ.shape[1]):
            # 将编码器输出 （enc_output） 传送至解码器
            predictions, dec_hidden, _ = model_params.decoder.call(dec_input, dec_hidden, enc_output)

            loss += loss_function(model_params, targ[:, t], predictions)

            # 使用教师强制
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = model_params.encoder.trainable_variables + model_params.decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    model_params.optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


def trainer(dataset_params, model_params, chk_params, dataset, epochs=100):
    for epoch in range(epochs):
        start = time.time()

        enc_hidden = model_params.encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(dataset_params.steps_per_epoch)):
            batch_loss = train_step(dataset_params, model_params, inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        # 每 2 个周期（epoch），保存（检查点）一次模型
        if (epoch + 1) % 2 == 0:
            chk_params.checkpoint.save(file_prefix=chk_params.checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / dataset_params.steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


def evaluate(dataset_params, model_params, sentence):
    attention_plot = np.zeros((dataset_params.max_length_targ, dataset_params.max_length_inp))

    sentence = up.extract_ch(sentence)
    print("============================", sentence)

    inputs = [dataset_params.inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=dataset_params.max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, dataset_params.units))]
    enc_out, enc_hidden = model_params.encoder.call(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([dataset_params.targ_lang.word_index['<start>']], 0)

    for t in range(dataset_params.max_length_targ):
        predictions, dec_hidden, attention_weights = model_params.decoder.call(dec_input,
                                                                               dec_hidden,
                                                                               enc_out)

        # 存储注意力权重以便后面制图
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += dataset_params.targ_lang.index_word[predicted_id] + ' '

        if dataset_params.targ_lang.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # 预测的 ID 被输送回模型
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


# 注意力权重制图函数
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 15}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig("mygraph.png")
    plt.show()


def restore_model(chk_params):
    chk_params.checkpoint.restore(tf.train.latest_checkpoint(chk_params.checkpoint_dir))


def translate(dataset_params, model_params, sentence):
    result, sentence, attention_plot = evaluate(dataset_params, model_params, sentence)

    print('Input: %s' % sentence)
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))


def params_dump(params, params_path):
    if params is not None:
        with open(params_path, 'wb') as f:
            pickle.dump(params, f)


def params_load(params_path):
    with open(params_path, 'rb') as f:
        return pickle.load(f)


def model_visualization(model_params):
    tf.keras.utils.plot_model(model_params.encoder, to_file='a.png', show_shapes=True)


