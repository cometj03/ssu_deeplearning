from dataset import ptb

if __name__ == '__main__':
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    print(corpus)