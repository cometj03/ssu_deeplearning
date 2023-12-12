from cbow.simple_cbow import SimpleCBOW
from rnn.simple_rnn import SimpleRnnlm


class CBOW_RNN:
    def __init__(self, vocab_size, wordvec_size,
                 cbow_hidden_size, in_rnn_hidden_size, out_rnn_hidden_size):
        O, I, C = out_rnn_hidden_size, in_rnn_hidden_size, cbow_hidden_size

        self.rnn0 = SimpleRnnlm(vocab_size, wordvec_size, I)
        self.rnn1 = SimpleRnnlm(vocab_size, wordvec_size, I)
        self.cbow = SimpleCBOW(vocab_size, C)
        self.rnn_out = SimpleRnnlm(vocab_size, wordvec_size, O)

        layers = [self.rnn0, self.rnn1, self.cbow, self.rnn_out]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, contexts, target):
        pass

    def backward(self, dout=1):
        pass
