# https://github.com/ssuai/deep_learning_from_scratch_2/blob/master/ch03/simple_cbow.py
import numpy as np

from common.layers import MatMul, SoftmaxWithLoss


class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size, enable_loss_layer=True):
        V, H = vocab_size, hidden_size

        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        # 계층 생성
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss() if enable_loss_layer else None

        # 모든 가중치와 기울기를 리스트에 모은다.
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 인스턴스 변수에 단어의 분산 표현을 저장한다.
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[0])
        h1 = self.in_layer1.forward(contexts[1])
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        if self.loss_layer is None:
            return score
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout=1):
        if self.loss_layer is not None:
            dout = self.loss_layer.backward(dout)
        da = self.out_layer.backward(dout)
        da *= 0.5
        d1 = self.in_layer1.backward(da)
        d0 = self.in_layer0.backward(da)
        return d0, d1
