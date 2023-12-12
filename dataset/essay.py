import json
import os
import pickle

import konlpy
import numpy as np
from konlpy.tag import Kkma

konlpy.jvm.init_jvm(jvmpath=None, max_heap_size=2048)

vocab_file = "essay.vocab.pkl"

dataset_dir = "D:/dataset/024.에세이 글 평가 데이터/01.데이터"
current_dir = os.path.dirname(os.path.abspath(__file__))

essay_dir = {
    "train": "라벨링데이터/TL_글짓기/글짓기",
    "valid": "라벨링데이터/VL_글짓기/글짓기"
}
key_dir = {
    "train": "1.Training",
    "valid": "2.Validation",
}
text_file_name = {
    "train": "essay.train.txt",
    "valid": "essay.valid.txt",
}
corpus_file_name = {
    "train": "essay_corpus.train.npy",
    "valid": "essay_corpus.valid.npy",
}


def _load_text(data_type: str = "train"):
    file_path = current_dir + "/" + text_file_name[data_type]
    if os.path.exists(file_path):
        return
    data_folder = dataset_dir + "/" + key_dir[data_type] + "/" + essay_dir[data_type]

    # 파일 이름에 '고등'이 들어간 파일 경로만 선택
    essay_file_paths = [data_folder + "/" + f for f in os.listdir(data_folder) if "고등" in f]

    print(data_folder + " 내의 파일을 읽는 중...")

    strs = []
    for path in essay_file_paths:
        with open(path, encoding="utf-8") as json_file:
            essay_dict = json.load(json_file)

        s = essay_dict["paragraph"][0]["paragraph_txt"]
        strs.append(s)
    total_str = "".join(strs)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(total_str)


def load_data(data_type='train', renew=False):
    '''
        :param data_type: 데이터 유형: 'train' or 'valid'
        :return:
    '''

    vocab_path = current_dir + '/' + vocab_file
    file_path = current_dir + '/' + text_file_name[data_type]
    corpus_file_path = current_dir + '/' + corpus_file_name[data_type]

    if not renew and os.path.exists(vocab_path) and os.path.exists(corpus_file_path):
        with open(vocab_path, 'rb') as f:
            word_to_id, id_to_word = pickle.load(f)
        corpus = np.load(corpus_file_path)
        return corpus, word_to_id, id_to_word

    _load_text(data_type)

    word_to_id = {}
    id_to_word = {}

    print("토크나이징 중")
    kkma = Kkma()
    words = open(file_path, encoding="utf-8").read().replace('#@문장구분#', '<eos>').strip()
    words = kkma.morphs(words)

    for i, word in enumerate(words):
        if word not in word_to_id:
            tmp_id = len(word_to_id)
            word_to_id[word] = tmp_id
            id_to_word[tmp_id] = word

    with open(vocab_path, 'wb') as f:
        pickle.dump((word_to_id, id_to_word), f)

    corpus = np.array([word_to_id[w] for w in words])
    np.save(corpus_file_path, corpus)

    print(f"어휘 개수 : {len(word_to_id)}")

    return corpus, word_to_id, id_to_word


if __name__ == '__main__':
    corpus, word_to_id, id_to_word = load_data(renew=True)
    print(f"corpus length : {len(corpus)}")
    print(corpus)
    print(word_to_id)
    print(id_to_word)
