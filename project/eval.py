# coding: utf-8
import pickle

from common.util import most_similar

pkl_file = "essay_cbow_params.pkl"
# pkl_file = 'cbow_params.pkl'
# pkl_file = 'skipgram_params.pkl'

if __name__ == '__main__':
    with open(pkl_file, 'rb') as f:
        params = pickle.load(f)
        word_vecs = params['word_vecs']
        word_to_id = params['word_to_id']
        id_to_word = params['id_to_word']

    # 가장 비슷한(most similar) 단어 뽑기
    querys = ["나", "어머니", "핸드폰"]
    for query in querys:
        most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

    # 유추(analogy) 작업
    # print('-' * 50)
    # analogy('king', 'man', 'queen', word_to_id, id_to_word, word_vecs)
    # analogy('take', 'took', 'go', word_to_id, id_to_word, word_vecs)
    # analogy('car', 'cars', 'child', word_to_id, id_to_word, word_vecs)
    # analogy('good', 'better', 'bad', word_to_id, id_to_word, word_vecs)
