import numpy as np

from cbow.simple_cbow import SimpleCBOW
from rnn.seq2seq import Decoder, Encoder
from common.time_layers import TimeRNN, TimeEmbedding, TimeSoftmaxWithLoss, TimeAffine


class CBOW_RNN:
    def __init__(self, vocab_size, wordvec_size,
                 cbow_hidden_size, in_rnn_hidden_size, decoder_hidden_size):
        IH, CH = in_rnn_hidden_size, cbow_hidden_size
        V, D = vocab_size, wordvec_size
        rn = np.random.randn

        # in_rnn 가중치 초기화
        in_embed_W = (rn(V, D) / 100).astype('f')
        in_rnn_Wx = (rn(D, IH) / np.sqrt(D)).astype('f')
        in_rnn_Wh = (rn(IH, IH) / np.sqrt(IH)).astype('f')
        in_rnn_b = np.zeros(IH).astype('f')

        # self.word_vecs = word_vecs # embedding 대신 word_vec 사용
        self.encoder0 = Encoder(V, D, IH)
        self.encoder1 = Encoder(V, D, IH)
        self.cbow = SimpleCBOW(IH, CH, enable_loss_layer=False)
        self.decoder = Decoder(vocab_size, wordvec_size, decoder_hidden_size)

        self.softmax = TimeSoftmaxWithLoss()

        layers = [self.encoder0, self.encoder1, self.cbow, self.decoder]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, contexts, target):
        """

        :param contexts: 주변 문장 (타입: list[np.ndarray])
        :param target: 타겟 문장 (타입: np.ndarray)
        :return:
        """
        xs0, xs1 = contexts[:, 0, :], contexts[:, 1, :]
        xs0 = self.encoder0.forward(xs0)
        xs1 = self.encoder1.forward(xs1)

        d = self.cbow.forward(np.array([xs0, xs1]), None)

        decoder_xs, decoder_ts = target[:, :-1], target[:, 1:]
        score = self.decoder.forward(decoder_xs, d)
        loss = self.softmax.forward(score, decoder_ts)
        return loss


    def backward(self, dout=1):
        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        d0, d1 = self.cbow.backward(dh)
        d1 = self.encoder1.backward(d1)
        d0 = self.encoder0.backward(d0)
        return d0, d1


