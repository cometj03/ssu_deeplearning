import pickle

import numpy as np

from dataset import essay
from project.cbow_with_rnn import CBOW_RNN
from project.util import create_context_target_sentences2


def save_params(pp, file_name=None):
    params = [p.astype(np.float16) for p in pp]

    with open(file_name, 'wb') as f:
        pickle.dump(params, f)

if __name__ == '__main__':
    wordvec_size = 100
    cbow_hidden_size = 30
    encoder_hidden_size = 100
    decoder_hidden_size = 100

    # 데이터 읽기
    corpus, word_to_id, id_to_word = essay.load_data('train')
    vocab_size = len(word_to_id)

    eos = word_to_id["##"]  # 문장 구분 문자
    contexts, target = create_context_target_sentences2(corpus, eos)

    pkl_file = "train_params.pkl"
    pkl_file2 = "train_params2.pkl"
    # with open(pkl_file, 'rb') as f:
    #     dic = pickle.load(f)
    # pretrained_param = dic["params"]

    model = CBOW_RNN(vocab_size, wordvec_size)
    model.load_params(pkl_file2)

    ctx = np.array([contexts[100]])
    first_text = [id_to_word[t] for t in ctx[0, 0, :] if t != eos]
    second_text = [id_to_word[t] for t in ctx[0, 1, :] if t != eos]

    text = model.generate(ctx, eos, 10)
    text = [id_to_word[t] for t in text if t != eos]

    print(first_text)
    print(second_text)
    print(text)