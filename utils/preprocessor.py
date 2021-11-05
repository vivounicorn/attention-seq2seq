import opencc
import unicodedata
import re
import jieba
import tensorflow as tf
import io


def t2s_converter(sentence):
    cvt = opencc.OpenCC('t2s')
    return cvt.convert(sentence)


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def find_first_chinese(s):
    han_str = re.compile('[\u4e00-\u9fff]+').findall(s)
    if len(han_str) <= 0:
        return None

    return han_str[0]


def extract_ch_en(sen, is_self=False):
    idx = sen.find('CC-BY')
    if idx == -1:
        return None

    w = unicode_to_ascii(sen[:idx].lower().strip())

    # 在单词与跟在其后的标点符号之间插入一个空格
    # 例如： "he is a boy." => "he is a boy ."
    # 参考：https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    w = w.rstrip().strip()

    # 给句子加上开始和结束标记
    # 以便模型知道何时开始和结束预测
    han = find_first_chinese(w)
    idx = w.find(han)

    if idx == -1:
        return None
    han_str = w[idx:]
    en_str = '<start> ' + w[:idx].strip() + ' <end>'
    if not is_self:
        w = [en_str, '<start> ' + chinese_words_cut(t2s_converter(han_str)) + ' <end>']
    else:
        w = ['<start> ' + chinese_words_cut(t2s_converter(han_str)) + ' <end>', '<start> ' + chinese_words_cut(t2s_converter(han_str)) + ' <end>']
    return w


def extract_ch(sen):

    w = unicode_to_ascii(sen.lower().strip())

    # 在单词与跟在其后的标点符号之间插入一个空格
    # 例如： "he is a boy." => "he is a boy ."
    # 参考：https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    w = w.rstrip().strip()

    # 给句子加上开始和结束标记
    # 以便模型知道何时开始和结束预测
    han = find_first_chinese(w)
    idx = w.find(han)

    if idx == -1:
        return None
    han_str = w[idx:]
    w = '<start> ' + chinese_words_cut(t2s_converter(han_str)) + ' <end>'
    return w


def chinese_words_cut(sentence):
    return " ".join(jieba.cut(sentence, cut_all=False))


# 1. 去除重音符号
# 2. 清理句子
# 3. 返回这样格式的单词对：[CHINESE, ENGLISH]
def create_dataset(path, num_examples, is_self=False):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [extract_ch_en(l, is_self) for l in lines[:num_examples]]
    print('here:', word_pairs[0], '\n')

    return zip(*word_pairs)


def max_length(tensor):
    return max(len(t) for t in tensor)


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')

    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    # 创建清理过的输入输出对
    targ_lang, inp_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


def convert(lang, tensor):
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, lang.index_word[t]))

