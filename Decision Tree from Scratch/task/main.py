import pandas as pd

class Node:

  def __init__(self):
    # class initialization
    self.left = None
    self.right = None
    self.term = False
    self.label = None
    self.feature = None
    self.value = None

  def set_split(self, feature, value):
    # this function saves the node splitting feature and its value
    self.feature = feature
    self.value = value

  def set_term(self, label):
    # if the node is a leaf, this function saves its label
    self.term = True
    self.label = label

  def __str__(self):
      if self.term:
          return f'Leaf = {self.label}'
      else:
          return f'Node split by {self.feature} = {self.value}:\n {self.left} {self.right}'

def gini(lst):
    n = len(lst)
    vals = set(lst)
    sum = 0
    for v in vals:
        sum += (lst.count(v) / n) ** 2
    return 1 - sum

def gini_w(lst1, lst2):
    n1 = len(lst1)
    n2 = len(lst2)
    wg = (n1 * gini(lst1) + n2 * gini(lst2)) / (n1 + n2)
    return wg

def is_leaf(X, y):
    if X.shape[0] <= 1 or gini(y.to_list()) == 0:
        return True
    return all(map(lambda x: len(set(x)) == 1, X.values.T))

def chose_split(X, y):
    g_min = 1
    feature = None
    f_value = None
    split_1, split_2 = None, None
    for col_name, values in X.iteritems():
        vals = values.unique()
        for v in vals:
            idx_1 = X.index[X[col_name] == v].tolist()
            idx_2 = X.index[X[col_name] != v].tolist()
            wg = gini_w(y.iloc[idx_1].tolist(), y.iloc[idx_2].tolist())
            if wg < g_min:
                g_min = wg
                feature = col_name
                f_value = v
                split_1, split_2 = idx_1, idx_2
    return feature, f_value, split_1, split_2

def split_df(node, X, y):
    if is_leaf(X, y):
        node.set_term(y[0])
        return
    feature, f_value, split_1, split_2 = chose_split(X, y)
    node.set_split(feature, f_value)
    print(f'Made split: {node.feature} is {node.value}')

    node.left = Node()
    node.right = Node()

    left_X = X.iloc[split_1].reset_index(drop=True)
    right_X = X.iloc[split_2].reset_index(drop=True)
    left_y = y.iloc[split_1].reset_index(drop=True)
    right_y = y.iloc[split_2].reset_index(drop=True)

    split_df(node.left, left_X, left_y)
    split_df(node.right, right_X, right_y)

def stage3():
    fn = input()
    # fn = 'test/data_stage3.csv'
    df = pd.read_csv(fn, index_col=0)
    X = df.iloc[:, :-1]
    y = df['Survived']
    root = Node()
    split_df(root, X, y)

stage3()
