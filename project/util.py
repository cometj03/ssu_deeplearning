import numpy as np


def split(arr: np.ndarray, sep: int):
    """리스트를 특정 숫자(sep)를 기준으로 나눕니다.

    :param arr:
    :param sep:
    :return:
    """
    indices = np.where(arr == sep)[0].tolist()
    indices = [-1] + indices + [len(arr)]
    res = []
    for t in range(1, len(indices)):
        i, j = indices[t - 1] + 1, indices[t]
        if i == j:
            continue
        res.append(arr[i:j])
    return res


def create_context_target_sentences(corpus, eos_id, window_size=1):
    """문장 단위의 맥락과 타깃 생성

    :param corpus: 말뭉치(단어 ID 목록)
    :param eos_id: eos(end of sentence)를 나타내는 아이디
    :param window_size: 윈도우 크기(윈도우 크기가 1이면 타깃 단어 좌우 한 단어씩이 맥락에 포함)
    :return:
    """
    corpus = split(corpus, eos_id)
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)
    return contexts, target


if __name__ == '__main__':
    ctx, target = create_context_target_sentences(np.array([1, 2, 0, 3, 4, 0, 2, 3, 0, 2, 2, 3, 1, 0]), 0)
    print(ctx)
    print(target)
