import numpy as np
from konlpy.tag import Kkma

if __name__ == '__main__':
    s = u"익숙함에 속아 소중함을 잊지말자라는 명언은 매우 유명한 명언이다.#@문장구분# 난 이 명언의 내용과 같은 것을 겪은 적이있다.#@문장구분#"

    kkma = Kkma()
    print(kkma.morphs(s))

