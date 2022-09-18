import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

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

class DecisionTree:
    def __init__(self, leaf_size=1, num_list=[]):
        self.root = Node()
        self.leaf_size = leaf_size
        self.num_list = num_list

    def fit(self, X, y):
        self._run_split(self.root, X, y)

    def predict(self, X):
        y_pred = []
        for i, x in X.iterrows():
            print(f'Prediction for sample # {i}')
            pred = self._chk_sample(self.root, x)
            y_pred.append(pred)
        return y_pred

    def _chk_sample(self, node, x):
        if node.term:
            print(f'   Predicted label: {node.label}')
            return node.label
        print(f'   Considering decision rule on feature {node.feature} with value {node.value}')
        if node.feature in self.num_list:
            if x[node.feature] <= node.value:
                return self._chk_sample(node.left, x)
            else:
                return self._chk_sample(node.right, x)
        else:
            if x[node.feature] == node.value:
                return self._chk_sample(node.left, x)
            else:
                return self._chk_sample(node.right, x)

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
        if X.shape[0] <= self.leaf_size or self._gini(y.to_list()) == 0:
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
                if col_name in self.num_list:
                    idx_1 = X.index[X[col_name] <= v].tolist()
                    idx_2 = X.index[X[col_name] > v].tolist()
                else:
                    idx_1 = X.index[X[col_name] == v].tolist()
                    idx_2 = X.index[X[col_name] != v].tolist()
                wg = self._gini_w(y.iloc[idx_1].tolist(), y.iloc[idx_2].tolist())
                if wg < g_min:
                    g_min = wg
                    feature = col_name
                    f_value = v
                    split_1, split_2 = idx_1, idx_2
        return g_min, feature, f_value, split_1, split_2

    def _run_split(self, node, X, y):
        if self._is_leaf(X, y):
            node.set_term(y.value_counts().idxmax())
            return
        gmi, feature, f_value, split_1, split_2 = self._chose_split(X, y)
        node.set_split(feature, f_value)
        print(f'Made split: {node.feature} is {node.value}')

        node.left = Node()
        node.right = Node()

        left_X = X.iloc[split_1].reset_index(drop=True)
        right_X = X.iloc[split_2].reset_index(drop=True)
        left_y = y.iloc[split_1].reset_index(drop=True)
        right_y = y.iloc[split_2].reset_index(drop=True)

        self._run_split(node.left, left_X, left_y)
        self._run_split(node.right, right_X, right_y)

def stage8():
    fn = input()
    # fn = 'test/data_stage8_train.csv test/data_stage8_test.csv'
    fn = fn.split()
    df = pd.read_csv(fn[0], index_col=0)
    X = df.iloc[:, :-1]
    y = df['Survived']
    df = pd.read_csv(fn[1], index_col=0)
    X_test = df.iloc[:]

    tree = DecisionTree(1, ['Age', 'Fare'])
    tree.fit(X, y)
    y_pred = tree.predict(X_test)

stage8()

