import pickle

from dataset import essay
from project.cbow_with_rnn import CBOW_RNN
from project.util import create_context_target_sentences2

if __name__ == '__main__':
    wordvec_size = 100
    cbow_hidden_size = 30
    encoder_hidden_size = 100
    decoder_hidden_size = 100
    batch_size = 30
    max_epoch = 1

    # 데이터 읽기
    corpus, word_to_id, id_to_word = essay.load_data('train')
    vocab_size = len(word_to_id)

    eos = word_to_id["##"]  # 문장 구분 문자
    contexts, target = create_context_target_sentences2(corpus, eos)

    pkl_file = "train_params.pkl"
    with open(pkl_file, 'rb') as f:
        dic = pickle.load(f)
    pretrained_param = dic["params"]

    model = CBOW_RNN(vocab_size, wordvec_size, pretrained_param=pretrained_param)
    text = model.generate(contexts[0], eos, 10)
    test = [id_to_word[t] for t in text]
    print(text)