from functools import reduce


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

node = input().split(' ')
node1 = input().split(' ')
node2 = input().split(' ')

g1 = gini(node)
g2 = gini_w(node1, node2)

print(round(g1, 2), round(g2, 2))
