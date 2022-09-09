import pandas as pd

def gi(a, b):
    p1 = a / (a + b)
    p2 = b / (a + b)
    return 1 - p1**2 - p2**2

def gini(node):
    n = len(node)
    p0 = node.count('0') / n
    p1 = node.count('1') / n
    g = 1 - p0 ** 2 - p1 ** 2
    return g

def gini_w(node1, node2):
    n1 = len(node1)
    n2 = len(node2)
    wg = (n1 * gini(node1) + n2 * gini(node2)) / (n1 + n2)
    return wg

def cnt(cat, tag, v):
    n, n0, n1 = 0, 0, 0
    for i in range(len(cat)):
        if cat[i] == v:
            n += 1
            if tag[i] == 0:
                n0 += 1
            else:
                n1 += 1
    return n, n0, n1

def gini_cat(cat, tag):
    num = len(cat)
    vals = set(cat)
    g_w = 0
    g_min = 1
    v_min = 0
    for v in vals:
        n, n0, n1 = cnt(cat, tag, v)
        g = gi(n0, n1)
        if g < g_min:
            g_min = g
            v_min = v
        g_w += g * n / num
    # print('g_w, v_min=', g_w, v_min)
    return g_w, v_min

def split(nod, v):
    nod1 = [i for i in range(len(nod)) if nod[i] == v]
    nod2 = [i for i in range(len(nod)) if nod[i] != v]
    return nod1, nod2

def stage1():
    node = input().split(' ')
    node1 = input().split(' ')
    node2 = input().split(' ')
    g1 = gini(node)
    g2 = gini_w(node1, node2)
    print(round(g1, 2), round(g2, 2))

def stage2():
    fn = input()
    # fn = 'test/data_stage2.csv'
    df = pd.read_csv(fn)
    tag = df['Survived']
    g_min = 1
    v_min = 0
    i_min = 0
    for i in range(1, df.shape[1] - 1):
        cat = df.iloc[:, i]
        g, v = gini_cat(cat, tag)
        if g < g_min:
            g_min = g
            v_min = v
            i_min = i
    nod = df.iloc[:, i_min]
    nod1, nod2 = split(nod, v_min)
    # print('G=', round(g_min, 4), 'cat=', df.columns[i_min], 'v=', v_min)
    print(round(g_min, 4), df.columns[i_min], v_min, nod1, nod2)


stage2()
