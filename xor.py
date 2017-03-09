import time

# cell은 W와 b를 가리키고, x는 0과 1의 조합
# x는 (0, 0), (0, 1), (1, 0), (1, 1)의 4가지 가능.
def check(cell, x):
    # 첫 번째 logistic 계산할 때 결과가 None이 되는 경우 존재
    if None in x:
        return None

    v = cell[0]*x[0] + cell[1]*x[1] + cell[2]
    # print(cell, x, v)

    # 0을 어떻게 처리할지 몰라 이번 코드에서는 제외
    if v == 0:
        return None

    if v < 0:
        return 0

    return 1


# xor 연산의 결과와 같은지 검사
def xor(cell, s1, s2):
    return [check(cell, (s1[i], s2[i])) for i in range(4)] == [0,1,1,0]


# 같은 패턴만 찾아내기 위한 함수. 패턴 구분은 0과 1의 조합으로 처리
def include(results, new):
    for _, _, av, bv, _ in results:
        if av == new[-2] and bv == new[-1]:
            return True

    return False


start = time.time()

# 리스트 컴프리헨션(comprehension)
a1 = [(i, j , k) for i in range(-10, 10, 3) for j in range(-10, 10, 3) for k in range(-10, 11, 4)]
a2 = [[check(i, x) for x in [(0,0), (0,1), (1,0), (1,1)]] for i in a1]

b1 = [(i, j , k) for i in range(-10, 10, 3) for j in range(-10, 10, 3) for k in range(-10, 11, 4)]
b2 = [[check(i, x) for x in [(0,0), (0,1), (1,0), (1,1)]] for i in b1]

c1 = [(i, j , k) for i in range(-10, 10, 3) for j in range(-10, 10, 3) for k in range(-10, 11, 4)]

# 생각없이 코딩한 3차원 반복문. 오랜만에 써본다.
results = []
for i, av in enumerate(a2):
    for j, bv in enumerate(b2):
        for k in c1:
            if xor(k, av, bv) == True:
                new = [a1[i], b1[j], av, bv]

                # 다른 패턴인 경우에만 추가. 패턴 결과는 출력을 보도록 한다.
                if include(results, new) == False:
                    new.append(k)
                    results.append(new)

print('elapsed :', time.time()-start)
for i in results:
    print(i)
