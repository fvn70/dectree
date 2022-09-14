import pandas as pd

class DecisionTree:
    def __init__(self, root):
        self.root = root
        self.leaf_size = 1

    def _gini(self, lst: list):
        n = len(lst)
        vals = set(lst)
        sum = 0
        for v in vals:
            sum += (lst.count(v) / n) ** 2
        return 1 - sum

    def _gini_w(self, lst1, lst2):
        n1 = len(lst1)
        n2 = len(lst2)
        wg = (n1 * self._gini(lst1) + n2 * self._gini(lst2)) / (n1 + n2)
        return wg

    def _is_leaf(self, X, y):
        if X.shape[0] <= 1 or self._gini(y.to_list()) == 0:
            return True
        return all(map(lambda x: len(set(x)) == 1, X.values.T))

    def _chose_split(self, X, y):
        g_min = 1
        feature = None
        f_value = None
        split_1, split_2 = None, None
        for col_name, values in X.iteritems():
            vals = values.unique()
            for v in vals:
                idx_1 = X.index[X[col_name] == v].tolist()
                idx_2 = X.index[X[col_name] != v].tolist()
                wg = self._gini_w(y.iloc[idx_1].tolist(), y.iloc[idx_2].tolist())
                if wg < g_min:
                    g_min = wg
                    feature = col_name
                    f_value = v
                    split_1, split_2 = idx_1, idx_2
        return feature, f_value, split_1, split_2

    def fit(self, node, X, y):
        if self._is_leaf(X, y):
            node.set_term(y[0])
            return
        feature, f_value, split_1, split_2 = self._chose_split(X, y)
        node.set_split(feature, f_value)
        print(f'Made split: {node.feature} is {node.value}')

        node.left = Node()
        node.right = Node()

        left_X = X.iloc[split_1].reset_index(drop=True)
        right_X = X.iloc[split_2].reset_index(drop=True)
        left_y = y.iloc[split_1].reset_index(drop=True)
        right_y = y.iloc[split_2].reset_index(drop=True)

        self.fit(node.left, left_X, left_y)
        self.fit(node.right, right_X, right_y)

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



def stage4():
    fn = input()
    # fn = 'test/data_stage4.csv'
    df = pd.read_csv(fn, index_col=0)
    X = df.iloc[:, :-1]
    y = df['Survived']
    root = Node()
    tree = DecisionTree(root)
    tree.fit(root, X, y)

stage4()
