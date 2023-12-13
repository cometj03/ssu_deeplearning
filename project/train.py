# coding: utf-8
import pickle

import numpy as np

from cbow.cbow import CBOW
from cbow_with_rnn import CBOW_RNN
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import create_contexts_target
from dataset import essay
from util import create_context_target_sentences


if __name__ == '__main__':
    # cbow_rnn 학습
    # 하이퍼파라미터 설정
    window_size = 1
    wordvec_size = 100
    cbow_hidden_size = 30
    in_rnn_hidden_size = 100
    out_rnn_hidden_size = 100
    batch_size = 10
    max_epoch = 1

    # 데이터 읽기
    corpus, word_to_id, id_to_word = essay.load_data('train')
    vocab_size = len(word_to_id)

    eos = word_to_id["eos"]
    contexts, target = create_context_target_sentences(corpus, eos, window_size)

    print(1)

    # 모델 등 생성
    model = CBOW_RNN(vocab_size, wordvec_size, cbow_hidden_size, in_rnn_hidden_size, out_rnn_hidden_size)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    print(2)

    # 학습 시작
    trainer.fit(contexts, target, max_epoch, batch_size)
    trainer.plot()

    print(3)

    # 나중에 사용할 수 있도록 필요한 데이터 저장
    # word_vecs = model.word_vecs
    # params = {}
    # params['word_vecs'] = word_vecs.astype(np.float16)
    # params['word_to_id'] = word_to_id
    # params['id_to_word'] = id_to_word
    # # pkl_file = 'cbow_params.pkl'  # or 'skipgram_params.pkl'
    # pkl_file = "essay_cbow_params.pkl"
    # with open(pkl_file, 'wb') as f:
    #     pickle.dump(params, f, -1)



def cbow_train():
    # 하이퍼파라미터 설정
    window_size = 5
    hidden_size = 100
    batch_size = 100
    max_epoch = 2

    # 데이터 읽기
    corpus, word_to_id, id_to_word = essay.load_data('train')
    vocab_size = len(word_to_id)

    contexts, target = create_contexts_target(corpus, window_size)

    # 모델 등 생성
    model = CBOW(vocab_size, hidden_size, window_size, corpus)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    # 학습 시작
    trainer.fit(contexts, target, max_epoch, batch_size)
    trainer.plot()

    # 나중에 사용할 수 있도록 필요한 데이터 저장
    word_vecs = model.word_vecs
    params = {}
    params['word_vecs'] = word_vecs.astype(np.float16)
    params['word_to_id'] = word_to_id
    params['id_to_word'] = id_to_word
    # pkl_file = 'cbow_params.pkl'  # or 'skipgram_params.pkl'
    pkl_file = "essay_cbow_params.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump(params, f, -1)
