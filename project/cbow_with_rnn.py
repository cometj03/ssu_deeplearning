import numpy as np

from cbow.simple_cbow import SimpleCBOW
from rnn.seq2seq import Decoder, Encoder
from common.time_layers import TimeRNN, TimeEmbedding, TimeSoftmaxWithLoss, TimeAffine


class CBOW_RNN:
    def __init__(self, vocab_size, wordvec_size,
                 cbow_hidden_size=30, encoder_hidden_size=100, decoder_hidden_size=100,
                 pretrained_param=None):
        IH, CH = encoder_hidden_size, cbow_hidden_size
        V, D = vocab_size, wordvec_size

        # self.word_vecs = word_vecs # embedding 대신 word_vec 사용
        self.encoder0 = Encoder(V, D, IH)
        self.encoder1 = Encoder(V, D, IH)
        self.cbow = SimpleCBOW(IH, CH, enable_loss_layer=False)
        self.decoder = Decoder(V, D, decoder_hidden_size)

        self.softmax = TimeSoftmaxWithLoss()

        layers = [self.encoder0, self.encoder1, self.cbow, self.decoder]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        if pretrained_param is not None:
            for i, param in enumerate(self.params):
                param[...] = pretrained_param[i]

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

    def generate(self, contexts, start_id, sample_size):
        xs0, xs1 = contexts[:, 0, :], contexts[:, 1, :]
        xs0 = self.encoder0.forward(xs0)
        xs1 = self.encoder1.forward(xs1)

        h = self.cbow.forward(np.array([xs0, xs1]), None)
        return self.decoder.generate(h, start_id, sample_size)