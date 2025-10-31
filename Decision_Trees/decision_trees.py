import pandas as pd
import numpy as np
import json
import os

# Tắt cảnh báo của pandas một cách an toàn
pd.options.mode.chained_assignment = None  # default='warn'

# Thử import graphviz, nếu không có sẽ báo lỗi thân thiện
try:
    from graphviz import Digraph
except ImportError:
    print("Thư viện 'graphviz' chưa được cài đặt.")
    print("Vui lòng chạy: pip install graphviz")
    print("Đồng thời, hãy đảm bảo bạn đã cài đặt phần mềm Graphviz trên hệ điều hành của mình.")
    print("Xem hướng dẫn tại: https://graphviz.org/download/")
    exit()

# =============================================================================
# CLASS ĐỊNH NGHĨA THUẬT TOÁN ID3
# =============================================================================
class ID3DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.tree = None

    # (Các phương thức _calculate_entropy, _calculate_information_gain, _find_best_split)
    def _calculate_entropy(self, y):
        if len(y) == 0: return 0
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    def _calculate_information_gain(self, X, y, feature):
        parent_entropy = self._calculate_entropy(y)
        unique_values = X[feature].unique()
        weighted_child_entropy = sum(
            (len(y[X[feature] == val]) / len(y)) * self._calculate_entropy(y[X[feature] == val])
            for val in unique_values
        )
        return parent_entropy - weighted_child_entropy

    def _find_best_split(self, X, y):
        if X.empty: return None
        gains = {feat: self._calculate_information_gain(X, y, feat) for feat in X.columns}
        best_feature = max(gains, key=gains.get)
        return best_feature if gains[best_feature] > 0 else None

    def _build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1:
            return {'leaf_value': y.iloc[0]}
        if len(y) < self.min_samples_split or depth >= self.max_depth:
            return {'leaf_value': y.mode()[0]}

        best_feature = self._find_best_split(X, y)
        if best_feature is None:
            return {'leaf_value': y.mode()[0]}

        tree = {'feature': best_feature, 'children': {}}
        for value in X[best_feature].unique():
            subset_indices = X[best_feature] == value
            subset_X = X.loc[subset_indices].drop(columns=[best_feature])
            subset_y = y.loc[subset_indices]
            if not subset_y.empty:
                tree['children'][str(value)] = self._build_tree(subset_X, subset_y, depth + 1)
        return tree

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    # (Các phương thức predict)
    def _predict_single(self, x, tree):
        if 'leaf_value' in tree: return tree['leaf_value']
        feature_name = tree['feature']
        if feature_name not in x: return self._get_most_common_leaf(tree)
        feature_value = str(x[feature_name])
        if feature_value in tree['children']:
            return self._predict_single(x, tree['children'][feature_value])
        else:
            return self._get_most_common_leaf(tree)

    def _get_most_common_leaf(self, tree_node):
        if 'leaf_value' in tree_node: return tree_node['leaf_value']
        leaf_values = [self._get_most_common_leaf(child) for child in tree_node['children'].values()]
        return max(set(leaf_values), key=leaf_values.count)

    def predict(self, X):
        return [self._predict_single(x, self.tree) for _, x in X.iterrows()]

    # --- PHƯƠNG THỨC MỚI ĐỂ VẼ CÂY ---
    def _visualize_recursive(self, dot, node, parent_id=None, edge_label=""):
        node_id = str(id(node))
        if 'leaf_value' in node:
            label = f"Class: {node['leaf_value']}"
            dot.node(node_id, label, shape='box', style='filled', fillcolor='lightgreen')
        else:
            feature = node['feature']
            label = f"{feature}?"
            dot.node(node_id, label, shape='ellipse', style='filled', fillcolor='lightblue')
            for value, child_node in node['children'].items():
                self._visualize_recursive(dot, child_node, node_id, edge_label=str(value))
        
        if parent_id:
            dot.edge(parent_id, node_id, label=edge_label)

    def visualize(self, filename='id3_tree'):
        if self.tree is None:
            print("Mô hình chưa được huấn luyện. Vui lòng gọi fit() trước.")
            return
        dot = Digraph(comment='ID3 Decision Tree')
        dot.attr(rankdir='TB', size='8,8')
        self._visualize_recursive(dot, self.tree)
        # Lưu file ảnh và file nguồn DOT
        dot.render(filename, format='png', view=False, cleanup=True)
        print(f"Hình ảnh cây quyết định ID3 đã được lưu vào file: {filename}.png")


# =============================================================================
# CLASS ĐỊNH NGHĨA THUẬT TOÁN CART
# =============================================================================
class CARTDecisionTree:
    def __init__(self, min_samples_split=2, max_depth=5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.tree = None

    # (Các phương thức tính toán và xây dựng cây)
    def _calculate_gini(self, y):
        if len(y) == 0: return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities**2)

    def _find_best_split(self, X, y):
        best_gini_reduction = 0
        best_split = None
        current_gini = self._calculate_gini(y)
        for feature in X.columns:
            for value in X[feature].unique():
                left_idx, right_idx = (X[feature] == value), (X[feature] != value)
                if not np.any(left_idx) or not np.any(right_idx): continue
                p_left = np.sum(left_idx) / len(y)
                weighted_gini = p_left * self._calculate_gini(y[left_idx]) + (1 - p_left) * self._calculate_gini(y[right_idx])
                gini_reduction = current_gini - weighted_gini
                if gini_reduction > best_gini_reduction:
                    best_gini_reduction = gini_reduction
                    best_split = {'feature': feature, 'value': value}
        return best_split

    def _build_tree(self, X, y, depth=0):
        if len(np.unique(y)) <= 1 or len(y) < self.min_samples_split or depth >= self.max_depth:
            return {'leaf_value': y.mode()[0] if not y.empty else None}
        best_split = self._find_best_split(X, y)
        if not best_split: return {'leaf_value': y.mode()[0]}
        
        feature, value = best_split['feature'], best_split['value']
        left_idx, right_idx = X[feature] == value, X[feature] != value
        
        tree = {
            'feature': feature, 'value': value,
            'left': self._build_tree(X[left_idx], y[left_idx], depth + 1),
            'right': self._build_tree(X[right_idx], y[right_idx], depth + 1)
        }
        return tree

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    # (Phương thức predict)
    def _predict_single(self, x, tree):
        if 'leaf_value' in tree: return tree['leaf_value']
        if x[tree['feature']] == tree['value']:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])

    def predict(self, X):
        return [self._predict_single(x, self.tree) for _, x in X.iterrows()]

    # --- PHƯƠNG THỨC ĐỂ VẼ CÂY ---
    def _visualize_recursive(self, dot, node, parent_id=None, edge_label=""):
        node_id = str(id(node))
        if 'leaf_value' in node:
            label = f"Class: {node['leaf_value']}"
            dot.node(node_id, label, shape='box', style='filled', fillcolor='lightcoral')
        else:
            feature, value = node['feature'], node['value']
            label = f"{feature} == {value}?"
            dot.node(node_id, label, shape='diamond', style='filled', fillcolor='skyblue')
            self._visualize_recursive(dot, node['left'], node_id, edge_label="True")
            self._visualize_recursive(dot, node['right'], node_id, edge_label="False")

        if parent_id:
            dot.edge(parent_id, node_id, label=edge_label)

    def visualize(self, filename='cart_tree'):
        if self.tree is None:
            print("Mô hình chưa được huấn luyện. Vui lòng gọi fit() trước.")
            return
        dot = Digraph(comment='CART Decision Tree')
        dot.attr(rankdir='TB', size='8,8')
        self._visualize_recursive(dot, self.tree)
        dot.render(filename, format='png', view=False, cleanup=True)
        print(f"Hình ảnh cây quyết định CART đã được lưu vào file: {filename}.png")


# =============================================================================
# CHƯƠNG TRÌNH CHÍNH
# =============================================================================

# --- BƯỚC 1: CHUẨN BỊ DỮ LIỆU ---
print("### BƯỚC 1: CHUẨN BỊ DỮ LIỆU ###")
data = {
    'Age': [41, 49, 37, 33, 27, 32, 28, 30, 38, 36, 29, 45, 35, 26, 50],
    'Attrition': ['Yes', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No'],
    'Department': ['Sales', 'R&D', 'R&D', 'R&D', 'R&D', 'R&D', 'Sales', 'R&D', 'R&D', 'Sales', 'R&D', 'Sales', 'R&D', 'R&D', 'Sales'],
    'JobSatisfaction': [4, 2, 3, 3, 2, 4, 1, 3, 3, 3, 1, 4, 2, 1, 3],
    'MaritalStatus': ['Single', 'Married', 'Single', 'Married', 'Married', 'Single', 'Single', 'Divorced', 'Single',
                       'Married', 'Single', 'Married', 'Single', 'Single', 'Divorced'],
    'OverTime': ['Yes', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No'],
    'WorkLifeBalance': [1, 3, 3, 3, 3, 2, 2, 3, 3, 2, 1, 3, 2, 1, 3]
}
df = pd.DataFrame(data)
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100], labels=['<=30', '31-40', '41-50', '51+'])
satisfaction_map = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
df['JobSatisfaction'] = df['JobSatisfaction'].map(satisfaction_map)
df['WorkLifeBalance'] = df['WorkLifeBalance'].map(satisfaction_map)

features = ['AgeGroup', 'Department', 'JobSatisfaction', 'MaritalStatus', 'OverTime', 'WorkLifeBalance']
target = 'Attrition'
X = df[features]
y = df[target]

# --- BƯỚC 2: MÔ HÌNH ID3 VÀ TRỰC QUAN HÓA ---
print("\n### BƯỚC 2: MÔ HÌNH ID3 ###")
id3_tree = ID3DecisionTree(min_samples_split=2, max_depth=4)
id3_tree.fit(X, y)
id3_tree.visualize('id3_attrition_tree') # Tạo file id3_attrition_tree.png

# --- BƯỚC 3: MÔ HÌNH CART VÀ TRỰC QUAN HÓA ---
print("\n### BƯỚC 3: MÔ HÌNH CART ###")
cart_tree = CARTDecisionTree(min_samples_split=2, max_depth=4)
cart_tree.fit(X, y)
cart_tree.visualize('cart_attrition_tree') # Tạo file cart_attrition_tree.png

# --- BƯỚC 4: DỰ ĐOÁN ---
print("\n### BƯỚC 4: DỰ ĐOÁN TRÊN DỮ LIỆU MỚI ###")
new_employee = pd.DataFrame([{'AgeGroup': '31-40', 'Department': 'Sales', 'JobSatisfaction': 'Low', 'MaritalStatus': 'Single', 'OverTime': 'Yes', 'WorkLifeBalance': 'Low'}])
prediction_id3 = id3_tree.predict(new_employee)
prediction_cart = cart_tree.predict(new_employee)
print(f"\nDự đoán của ID3: {prediction_id3[0]}")
print(f"Dự đoán của CART: {prediction_cart[0]}")