from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple

np.seterr(divide='ignore', invalid='ignore')

@dataclass
class Node:
    """Decision tree node with uplift-specific fields."""
    n_items: int
    ATE: float
    split_feat: int
    split_threshold: float
    left: 'Node' = None
    right: 'Node' = None

@dataclass
class UpliftTreeRegressor:
    def __init__(self, max_depth,
                 min_samples_leaf,
                 min_samples_leaf_treated,
                 min_samples_leaf_control):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_leaf_treated = min_samples_leaf_treated
        self.min_samples_leaf_control = min_samples_leaf_control

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> 'UpliftTreeRegressor':
        self.n_features_ = X.shape[1]
        self.tree_ = self._build(X, treatment, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict uplift scores for a given set of samples.

        Parameters:
        X (np.ndarray): Input data with shape (n_samples, n_features).

        Returns:
        np.ndarray: Predicted uplift scores for each sample.
        """
        uplift_scores = []

        for sample in X:
            node = self.tree_
            while node.left is not None and node.right is not None:
                if sample[node.split_feat] <= node.split_threshold:
                    node = node.left
                else:
                    node = node.right

            uplift_scores.append(node.ATE)

        return np.array(uplift_scores)

    def find_threshold_options(self, column_values):
        unique_values = np.unique(column_values)
        if len(unique_values) > 10:
            percentiles = np.percentile(
                column_values, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97]
            )
        else:
            percentiles = np.percentile(unique_values, [10, 50, 90])

        threshold_options = np.unique(percentiles)
        return threshold_options

    def calculate_delta_delta_p(self, X, y, treatment, feature, threshold):
        treated_mask = (treatment == 1) & (X[:, feature] <= threshold)
        control_mask = (treatment == 0) & (X[:, feature] <= threshold)

        treated_group = y[treated_mask]
        control_group = y[control_mask]

        uplift_treated_left = np.mean(treated_group) - np.mean(control_group)

        treated_mask = (treatment == 1) & (X[:, feature] > threshold)
        control_mask = (treatment == 0) & (X[:, feature] > threshold)

        treated_group = y[treated_mask]
        control_group = y[control_mask]

        uplift_treated_right = np.mean(treated_group) - np.mean(control_group)

        delta_delta_p = np.abs(uplift_treated_left - uplift_treated_right)

        return delta_delta_p

    def _best_trashold(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray, feature: int) -> Tuple[float, float]:
        best_threshold = None
        best_delta_delta_p = float('-inf')

        threshold_options = self.find_threshold_options(X[:, feature])

        for threshold in threshold_options:
            left_indices = X[:, feature] <= threshold
            right_indices = X[:, feature] > threshold
            left_treated = treatment[left_indices]
            right_treated = treatment[right_indices]
            left_control = treatment[left_indices] == 0
            right_control = treatment[right_indices] == 0

            if (
                np.sum(left_indices) < self.min_samples_leaf
                or np.sum(right_indices) < self.min_samples_leaf
                or np.sum(left_treated) < self.min_samples_leaf_treated
                or np.sum(right_treated) < self.min_samples_leaf_treated
                or np.sum(left_control) < self.min_samples_leaf_control
                or np.sum(right_control) < self.min_samples_leaf_control
            ):
                continue

            delta_delta_p = self.calculate_delta_delta_p(X, y, treatment, feature, threshold)
            if delta_delta_p > best_delta_delta_p:
                best_delta_delta_p = delta_delta_p
                best_threshold = threshold

        return best_threshold, best_delta_delta_p

    def _best_split(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> Tuple[int, float, float]:
        best_feature = None
        best_threshold = None
        best_delta_delta_p = float('-inf')

        for feature in range(self.n_features_):
            threshold, delta_delta_p = self._best_trashold(X=X, treatment=treatment, y=y, feature=feature)
            if delta_delta_p > best_delta_delta_p:
                best_delta_delta_p = delta_delta_p
                best_threshold = threshold
                best_feature = feature

        return best_feature, best_threshold, best_delta_delta_p


    def _build(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        n_samples = len(y)
        uplift = np.abs(np.mean(y[treatment == 1]) - np.mean(y[treatment == 0]))

        node = Node(
            n_items=n_samples,
            ATE=uplift,
            split_feat=None,
            split_threshold=0,
        )

        if (
            depth >= self.max_depth
            or n_samples <= self.min_samples_leaf
            or np.sum(treatment) < self.min_samples_leaf_treated
            or np.sum(treatment==0) < self.min_samples_leaf_control
            ):
            return node

        best_feature, best_threshold, best_delta_delta_p = self._best_split(X, treatment, y)

        node.split_feat = best_feature
        node.split_threshold = best_threshold

        if best_threshold is not None:
            left_mask =  (X[:, best_feature] <= best_threshold)
            right_mask = (X[:, best_feature] > best_threshold)

            X_left, treatment_left, y_left = X[left_mask], treatment[left_mask], y[left_mask]
            X_right, treatment_right, y_right = X[right_mask], treatment[right_mask], y[right_mask]

            node.left = self._build(X_left, treatment_left, y_left, depth + 1)
            node.right = self._build(X_right, treatment_right, y_right, depth + 1)

        return node

    def save_tree_to_txt(self, node, file, depth=0):
        if node is not None:
            # Print the current node
            file.write('\t' * depth + node.__class__.__name__ + '\n')
            file.write('\t' * (depth + 1) + f'n_items: {node.n_items}\n')
            file.write('\t' * (depth + 1) + f'ATE: {node.ATE}\n')
            file.write('\t' * (depth + 1) + f'split_feat: {node.split_feat}\n')
            file.write('\t' * (depth + 1) + f'split_threshold: {node.split_threshold}\n')

            if node.left is None and node.right is None:
                file.write('\t' * (depth + 1) + f'<leaf>\n')
                file.write('\t' * (depth + 2) + f'n_items: {node.n_items}\n')
                file.write('\t' * (depth + 2) + f'ATE: {node.ATE}\n')
                file.write('\t' * (depth + 2) + f'split_feat: {node.split_feat}\n')
                file.write('\t' * (depth + 2) + f'split_threshold: {node.split_threshold}\n')
            else:
                # Recursively write the left child node
                self.save_tree_to_txt(node.left, file, depth + 1)

                # Recursively write the right child node
                self.save_tree_to_txt(node.right, file, depth + 1)
