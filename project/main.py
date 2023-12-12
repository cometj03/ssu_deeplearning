from konlpy.tag import Kkma

if __name__ == '__main__':
    s = u'나는 밥을 먹었다. 이 사건을 통해 그 친구와 멀어지게 되었습니다.'
    kkma = Kkma()
    res = kkma.pos(s)
    print(res)
