import numpy as np
import pandas as pd
import json
from typing import Dict, Any, List, Tuple, Union, TypedDict, Optional, Literal
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,  f1_score, recall_score, precision_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier
import argparse
import benedict

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, column_or_1d
from sklearn.utils.multiclass import unique_labels, type_of_target


import dotenv
#TODO nice: add argparse to parse other .json's
#TODO nice: add feature recombinator
#TODO nice: option for a GradientBoostedClassifier followed by RuleFit

#TODO must: add pickle function or ONNX; https://onnx.ai/sklearn-onnx/supported.html,
#TODO must: add error handling in main functions
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Node:
    def __init__(self, feature=None, threshold=None, condition="less_than_or_equal",
                 left=None, right=None, value=None, pre_condition_value=None):
        self.feature = feature
        self.threshold = threshold
        self.condition = condition  # "less_than_or_equal", "greater_than", "less_than", "greater_than_or_equal"
        self.left = left
        self.right = right
        self.value = value
        self.pre_condition_value = pre_condition_value

class RuleNode:
    def __init__(self, name: str,
                 feature: Optional[str],
                 condition: Optional[Union[str, List[str]]],
                 threshold: Optional[float] = None,
                 class_counts: Optional[List[int]] = None,
                 ignore_after: Optional[List[str]] = None,
                 features_to_use_next: Optional[List[str]] = None,
                 pre_condition_value: Optional[Any] = None,
                 is_custom: bool = False,
                 samples: int = 0):
        self.name = self._validate_non_empty_string(name, "name")
        self.feature = self._validate_feature(feature)
        self.condition = self._validate_condition(condition)
        self.threshold = threshold
        self.class_counts = class_counts
        self.ignore_after = ignore_after or []
        self.features_to_use_next = features_to_use_next or []
        self.pre_condition_value = pre_condition_value
        self.children: List[RuleNode] = []
        self.probas = None
        self.coverage = None
        self.is_custom = is_custom
        self.samples = samples

    @staticmethod
    def _validate_non_empty_string(value: str, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
        return value

    @staticmethod
    def _validate_feature(feature: Optional[str]) -> Optional[str]:
        if feature is None:
            return None
        if not isinstance(feature, str) or not feature.strip():
            raise ValueError("feature must be None or a non-empty string")
        return feature.strip()

    @staticmethod
    def _validate_condition(condition: Union[str, List[str]]) -> Union[str, List[str]]:
        if condition is None:
            return None

        map_synonyms = {
            "lower_than": "less_than",
            "lower_than_or_equal": "less_than_or_equal",
            "lower_than_or_equal_to": "less_than_or_equal",
            "less_than_or_equal_to": "less_than_or_equal",
            "higher_than": "greater_than",
            "higher_than_or_equal": "greater_than_or_equal",
            "higher_than_or_equal_to": "greater_than_or_equal",
            "greater_than_or_equal_to": "greater_than_or_equal"
        }

        valid_conditions = ["less_than", "less_than_or_equal", "greater_than", "greater_than_or_equal"]

        def normalize_single_condition(cond: str) -> str:
            normalized_cond = cond.lower()
            if normalized_cond in valid_conditions:
                return normalized_cond
            if normalized_cond in map_synonyms:
                return map_synonyms[normalized_cond]
            raise ValueError(f"condition '{cond}' must be one of {valid_conditions} or their synonyms")

        if isinstance(condition, str):
            return normalize_single_condition(condition)
        elif isinstance(condition, list):
            return [normalize_single_condition(cond) for cond in condition]
        else:
            raise ValueError("condition must be either a string or a list of strings")

    def add_child(self, child: 'RuleNode', preserve_custom: bool = True) -> None:
        if preserve_custom:
            child.is_custom = self.is_custom
        self.children.append(child)

class LoadRules:
    def __init__(self, rules_file: str, name_map: Dict[str, str]=None):
        self.rules = self._load_rules(rules_file)
        self.fold_split_col = self.rules.get("fold_split_col")
        self.target_col = self.rules.get("target_col")
        self.ignore_cols = self.rules.get("ignore_cols", [])
        self.features_to_use = self.rules.get("features_to_use", [])

        if isinstance(name_map, dict):
            self.name_map = name_map
            try:
                self.fold_split_col = name_map[self.fold_split_col]
                self.target_col = name_map[self.target_col]
                self.ignore_cols = [name_map[col] for col in self.ignore_cols]
                self.features_to_use = [name_map[feature] for feature in self.features_to_use]
            except KeyError:
                raise KeyError("Name map must contain all columns used in the rules")
        else:
            self.name_map = None

        self.root = self._process_rules(self.rules["root"])

    def _name_mapper(self, name):
        if self.name_map is None:
            return name
        else:
            if name is None:
                return None
            else:
                return self.name_map[name]
    def _load_rules(self, rules_file: str) -> Dict[str, Any]:
        with open(rules_file, 'r') as f:
            rules = json.load(f)
        return rules

    def _process_rules(self, node: Dict[str, Any]) -> RuleNode:
        processed_node = RuleNode(
            name=node["name"],
            feature=self._name_mapper(node.get("feature")),
            condition=node.get("condition"),
            threshold=node.get("value"),  # Use 'value' as threshold for internal nodes
            class_counts=None,  # This will be set during tree building
            ignore_after=[self._name_mapper(c) for c in node.get("ignore_after", [])],
            features_to_use_next=[self._name_mapper(c) for c in node.get("features_to_use_next", [])],
            is_custom=True
        )

        if "pre_condition_value" in node:
            processed_node.pre_condition_value = node["pre_condition_value"]

        if "children" in node:
            for child in node["children"]:
                processed_node.add_child(self._process_rules(child))

        return processed_node

    def get_processed_rules(self) -> RuleNode:
        return self.root

class CustomDecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 custom_rules: Union[Dict[str, Any], RuleNode] = None,
                 criterion: str='gini', 
                 tot_max_depth: int = None,
                 max_depth: int = None,
                 min_samples_split: int = 10,
                 min_samples_leaf: int = 5,
                 random_state: int=None,
                 prune_threshold: float=0.9,
                 TargetMap: Dict[int,str]=None,
                 Tree_kwargs: Dict[str, Any] = None):
        '''
        Custom decision tree builder v2; this applies the custom tree first and then continues
        building trees from the leaves of the custom tree.

        :param custom_rules: Dictionary of custom rules or a RuleNode object
        :param criterion: the decision tree criterion
        :param tot_max_depth: total maximum depth of the decision tree
        :param max_depth: maximum depth of the decision tree
        :param min_samples_split: minimum number of samples required to split an internal node
        :param min_samples_leaf: minimum number of samples required to be at a leaf node
        :param random_state: the seed used by the random number generator
        :param prune_threshold: threshold for pruning the tree
        :param TargetMap: dictionary mapping target labels to their names
        :param Tree_kwargs: additional keyword arguments for the DecisionTreeClassifier
        '''
        super().__init__()
        self.custom_rules = custom_rules
        self.prune_threshold = prune_threshold
        self.tree_ = None
        self.feature_names_ = None
        self.enriched_rules = None
        self.features_to_consider = None
        self.classes_ = None
        self.n_classes_ = None    
        
        self.criterion = criterion
        self.max_depth = max_depth
        self.tot_max_depth = tot_max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.TargetMap = TargetMap
        
        self.Tree_kwargs = Tree_kwargs or {}
        if Tree_kwargs is {}:
            self.Tree_kwargs['criterion'] = criterion
            self.Tree_kwargs['max_depth'] = max_depth
            self.Tree_kwargs['min_samples_split'] = min_samples_split
            self.Tree_kwargs['min_samples_leaf'] = min_samples_leaf
            self.Tree_kwargs['random_state'] = random_state



    def get_params(self, deep=True):
        """Get parameters for this estimator.

        This method is overridden to explicitly list all parameters.
        """
        return {
            "custom_rules": self.custom_rules,
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
            "prune_threshold": self.prune_threshold
        }

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        # Store additional attributes set during fitting
        self.tree_ = None
        self.feature_names_ = X.columns.tolist()
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.features_to_consider = self._get_features_to_consider(X)
        logger.info("Starting fit method")
        try:
            # Check that X is a DataFrame
            if not isinstance(X, pd.DataFrame):
                raise ValueError("X must be a pandas DataFrame")

            # Validate X
            check_array(X, force_all_finite='allow-nan', ensure_2d=True, dtype=None)
            
            # Validate y
            y = column_or_1d(y, warn=True)
            check_array(y, ensure_2d=False, dtype=None)

            # Check that X and y have the same first dimension
            if X.shape[0] != y.shape[0]:
                raise ValueError("X and y must have the same number of samples")
            
            # Check that y is a valid target type
            target_type = type_of_target(y)
            if target_type not in ['binary', 'multiclass']:
                raise ValueError("Unknown label type: %r" % target_type)

            # Store the classes seen during fit
            self.classes_ = unique_labels(y)
            self.n_classes_ = len(self.classes_)
            self.feature_names_ = X.columns.tolist()

            logger.info(f"Number of features: {len(self.feature_names_)}")
            logger.info(f"Number of classes: {self.n_classes_}")

            # Extract features to consider for continuation
            self.features_to_consider = self._get_features_to_consider(X)
            logger.info(f"Features to consider: {self.features_to_consider}")

            # Create the initial tree structure based on custom rules
            if isinstance(self.custom_rules, dict):
                logger.info(f"Initial tree structure - Dictionary")
                self.tree_ = self._build_custom_tree(self.custom_rules['root'], X, y)
            elif isinstance(self.custom_rules, RuleNode):
                logger.info(f"Initial tree structure - RuleNode")
                self.tree_ = self.process_custom_rules(self.custom_rules, X, y)
            else:
                raise ValueError("custom_rules must be either a dictionary or a RuleNode object")

            logger.info("Initial tree structure created")

            # Continue building the tree with DecisionTreeClassifier at the leaves
            self._continue_tree_building(X, y)
            logger.info("Tree building completed")

            # Enrich custom tree with probabilities and coverage
            self.enriched_rules = self.enrich_custom_tree(X, y)
            logger.info("Tree enrichment completed")

            # Prune the tree if a threshold is set
            if self.prune_threshold is not None:
                self._prune_tree(self.tree_)
                logger.info("Tree pruning completed")

        except Exception as e:
            logger.error(f"An error occurred during fitting: {str(e)}", exc_info=True)
            raise

        return self
    def _get_matching_features(self, feature_list: List[str], all_features: List[str]) -> List[str]:
        """
        Find all features in all_features that contain any of the strings in feature_list.
        """
        self.features_to_use =[f for f in all_features if any(spec in f for spec in feature_list)]
        return self.features_to_use

    def _get_features_to_consider(self, X: pd.DataFrame) -> List[str]:
        all_features = X.columns.tolist()
        if isinstance(self.custom_rules, dict):
            if 'ignore_after' in self.custom_rules:
                ignore_features = set(self._get_matching_features(self.custom_rules['ignore_after'], all_features))
                return [f for f in all_features if f not in ignore_features]
            if 'features_to_use' in self.custom_rules:
                return self._get_matching_features(self.custom_rules['features_to_use'], all_features)
        elif isinstance(self.custom_rules, RuleNode):
            ignore_features = set(self._get_matching_features(self.custom_rules.ignore_after, all_features))
            if self.custom_rules.features_to_use_next:
                return self._get_matching_features(self.custom_rules.features_to_use_next, all_features)
            return [f for f in all_features if f not in ignore_features]
        return all_features

    def process_custom_rules(self, node: RuleNode, X: pd.DataFrame, y: np.ndarray) -> RuleNode:
        if node is None:
            return None

        node.samples = len(y)
        node.class_counts = np.bincount(y, minlength=self.n_classes_).tolist()

        if node.feature is None:
            # This is a leaf node
            return node

        # This is an internal node
        if node.condition == 'less_than':
            mask = X[node.feature] < node.threshold
        elif node.condition == 'less_than_or_equal':
            mask = X[node.feature] <= node.threshold
        elif node.condition == 'greater_than':
            mask = X[node.feature] > node.threshold
        elif node.condition == 'greater_than_or_equal':
            mask = X[node.feature] >= node.threshold
        else:
            raise ValueError(f"Unknown condition: {node.condition}")

        # Process children
        if not node.children:
            # If children don't exist, create them
            left_child = RuleNode(name="left_child", feature=None, condition=None,
                                  class_counts=None,  # Will be set in recursive call
                                  ignore_after=node.ignore_after,
                                  features_to_use_next=node.features_to_use_next,
                                  is_custom=True)
            right_child = RuleNode(name="right_child", feature=None, condition=None,
                                   class_counts=None,  # Will be set in recursive call
                                   ignore_after=node.ignore_after,
                                   features_to_use_next=node.features_to_use_next,
                                   is_custom=True)
            node.add_child(left_child)
            node.add_child(right_child)

        node.children[0] = self.process_custom_rules(node.children[0], X[mask], y[mask])
        node.children[1] = self.process_custom_rules(node.children[1], X[~mask], y[~mask])

        return node
    def _build_custom_tree(self, node: Dict[str, Any],
                           X: pd.DataFrame,
                           y: np.ndarray) -> RuleNode:
        if 'feature' not in node:
            # This is a leaf node
            return RuleNode(name="leaf", feature=None, condition=None,
                            value=np.bincount(y, minlength=self.n_classes_).tolist(),
                            samples=len(y),
                            is_custom=True)

        feature = self._get_matching_features([node['feature']], X.columns)[0]  # Get the first matching feature
        condition = node['condition']
        threshold = node['value']
        ignore_after = self._get_matching_features(node.get('ignore_after', []), X.columns)
        features_to_use_next = self._get_matching_features(node.get('features_to_use_next', []), X.columns)

        # Split the data
        if condition == 'less_than':
            left_mask = X[feature] < threshold
        elif condition == 'less_than_or_equal':
            left_mask = X[feature] <= threshold
        elif condition == 'greater_than':
            left_mask = X[feature] > threshold
        elif condition == 'greater_than_or_equal':
            left_mask = X[feature] >= threshold
        else:
            raise ValueError(f"Unknown condition: {condition}")

        right_mask = ~left_mask
        left_X, left_y = X[left_mask], y[left_mask]
        right_X, right_y = X[right_mask], y[right_mask]

        # Create the current node
        current_node = RuleNode(name=node['name'],
                                feature=feature, condition=condition, value=threshold,
                                ignore_after=ignore_after,
                                features_to_use_next=features_to_use_next,
                                is_custom=True,
                                samples=len(y))

        # Process children if they exist
        if 'children' in node and node['children']:
            for child in node['children']:
                pre_condition_value = child.get('pre_condition_value', True)
                child_node = self._build_custom_tree(child, left_X if pre_condition_value else right_X,
                                                     left_y if pre_condition_value else right_y)
                current_node.add_child(child_node, preserve_custom=True)

        # If a child is not set, create a leaf node
        if len(current_node.children) == 0:
            left_leaf = RuleNode(name="left_leaf", feature=None, condition=None,
                                            value=np.bincount(left_y,
                                            minlength=self.n_classes_).tolist(),
                                            is_custom=True,
                                            samples=len(left_y))
            right_leaf = RuleNode(name="right_leaf", feature=None, condition=None,
                                            value=np.bincount(right_y,
                                            minlength=self.n_classes_).tolist(),
                                            is_custom=True,
                                            samples=len(right_y))
            current_node.add_child(left_leaf, preserve_custom=True)
            current_node.add_child(right_leaf, preserve_custom=True)


        return current_node

    def _continue_tree_building(self, X: pd.DataFrame, y: np.ndarray):
        def build_subtree(node: RuleNode, node_X: pd.DataFrame, node_y: np.ndarray, depth: int = 0):
            if (len(node_X) == 0 or len(node_y) == 0) | (depth > self.tot_max_depth):
                logger.warning("No samples left for subtree. Making a leaf node.")
                node.feature = None
                node.condition = None
                node.class_counts = np.zeros(self.n_classes_).tolist()
                node.children = []
                node.is_custom = False
                return

            # Determine which features to use for subsequent splits
            if node.features_to_use_next is None or len(node.features_to_use_next) == 0:
                features_for_splits = [f for f in self.features_to_consider if
                                       f not in self._get_matching_features(node.ignore_after, node_X.columns)]
            else:
                features_for_splits = self._get_matching_features(node.features_to_use_next, node_X.columns)

            if not features_for_splits:
                logger.warning("No features available for splitting. Making a leaf node.")
                node.class_counts = np.bincount(node_y, minlength=self.n_classes_).tolist()
                node.is_custom = False
                return

            if node.feature is None:
                # This is a leaf node in the custom tree, continue with DecisionTreeClassifier
                subtree = DecisionTreeClassifier(**self.Tree_kwargs)
                subtree.fit(node_X[features_for_splits], node_y)

                if subtree.tree_.feature[0] != -2:  # -2 indicates a leaf in scikit-learn's implementation
                    node.feature = features_for_splits[subtree.tree_.feature[0]]
                    node.condition = 'less_than_or_equal'
                    node.threshold = subtree.tree_.threshold[0]
                    node.is_custom = False

                    left_node = RuleNode(name="left_child", feature=None, condition=None,
                                         class_counts=subtree.tree_.value[subtree.tree_.children_left[0]][0].tolist(),
                                         ignore_after=node.ignore_after,
                                         features_to_use_next=node.features_to_use_next,
                                         is_custom=False,
                                         samples=subtree.tree_.n_node_samples[subtree.tree_.children_left[0]])
                    right_node = RuleNode(name="right_child", feature=None, condition=None,
                                          class_counts=subtree.tree_.value[subtree.tree_.children_right[0]][0].tolist(),
                                          ignore_after=node.ignore_after,
                                          features_to_use_next=node.features_to_use_next,
                                          is_custom=False,
                                          samples=subtree.tree_.n_node_samples[subtree.tree_.children_right[0]])

                    node.add_child(left_node)
                    node.add_child(right_node)

                    left_mask = node_X[node.feature] <= node.threshold
                    build_subtree(node.children[0], node_X[left_mask], node_y[left_mask.values], depth + 1)
                    build_subtree(node.children[1], node_X[~left_mask], node_y[~left_mask.values], depth + 1)
                else:
                    # If the subtree is just a leaf, update the current node
                    node.class_counts = subtree.tree_.value[0][0].tolist()
                    node.is_custom = False
            else:
                # This is an internal node from the custom rules, process its children
                if node.condition in ['less_than', 'less_than_or_equal']:
                    mask = node_X[node.feature] <= node.threshold
                else:
                    mask = node_X[node.feature] > node.threshold

                # If children don't exist, create them
                if not node.children:
                    left_child = RuleNode(name="left_child", feature=None, condition=None,
                                          class_counts=np.bincount(node_y[mask], minlength=self.n_classes_).tolist(),
                                          ignore_after=node.ignore_after,
                                          features_to_use_next=node.features_to_use_next,
                                          is_custom=node.is_custom,
                                          samples=np.sum(mask))
                    right_child = RuleNode(name="right_child", feature=None, condition=None,
                                           class_counts=np.bincount(node_y[~mask], minlength=self.n_classes_).tolist(),
                                           ignore_after=node.ignore_after,
                                           features_to_use_next=node.features_to_use_next,
                                           is_custom=node.is_custom,
                                           samples=np.sum(~mask))
                    node.add_child(left_child)
                    node.add_child(right_child)

                # Continue building subtrees for both children
                build_subtree(node.children[0], node_X[mask], node_y[mask], depth + 1)
                build_subtree(node.children[1], node_X[~mask], node_y[~mask], depth + 1)

        # Start building subtrees from the root
        build_subtree(self.tree_, X, y)

    def _prune_tree(self, node: RuleNode):
        if node is None or not node.children:
            # This is a leaf node or an empty node, nothing to prune
            return False

        # Recursively prune children
        pruned_children = [self._prune_tree(child) for child in node.children]

        # If all children are leaves (after potential pruning), consider pruning this node
        if all(child.feature is None for child in node.children):
            node_proba = max(node.probas) if hasattr(node, 'probas') else 0

            if node_proba >= self.prune_threshold:
                # Prune this node (make it a leaf)
                print(f"Pruning node with feature: {node.feature}, condition: {node.condition}, value: {node.threshold}")
                node.feature = None
                node.condition = None
                # We keep the node's existing 'value', 'probas', and 'coverage' as they are already correct
                node.children = []
                return True  # Indicate that pruning occurred

        # Remove pruned children
        node.children = [child for child, pruned in zip(node.children, pruned_children) if not pruned]

        return False  # Indicate that no pruning occurred at this node

    def prune_tree(self):
        def prune_recursive(node):
            if node is None:
                return 0

            pruned_count = sum(prune_recursive(child) for child in node.children)

            if self._prune_tree(node):
                pruned_count += 1

            return pruned_count

        total_pruned = prune_recursive(self.tree_)
        print(f"Total nodes pruned: {total_pruned}")

    def print_tree_structure(self, node=None, depth=0):
        if node is None:
            node = self.tree_

        indent = "  " * depth
        if node.feature is None:
            print(f"{indent}Leaf: value={node.value}, probas={node.probas}, coverage={node.coverage}")
        else:
            print(f"{indent}{node.feature} {node.condition} {node.value}")
            print(f"{indent}probas={node.probas}, coverage={node.coverage}")
            for child in node.children:
                self.print_tree_structure(child, depth + 1)

    def predict(self, X):
        check_is_fitted(self)
        # Convert to DataFrame if it's a numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_)
        else:
            X = check_array(X, force_all_finite='allow-nan', ensure_2d=True, dtype=None)
            X = pd.DataFrame(X, columns=self.feature_names_)
        return self._predict(X)

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.tree_ is None:
            raise NotFittedError("This CustomDecisionTree instance is not fitted yet")

        def predict_sample(node: RuleNode, sample: pd.Series) -> int:
            if node.feature is None:
                return np.argmax(node.class_counts)
            if node.condition == 'less_than':
                go_left = sample[node.feature] < node.threshold
            elif node.condition == 'less_than_or_equal':
                go_left = sample[node.feature] <= node.threshold
            elif node.condition == 'greater_than':
                go_left = sample[node.feature] > node.threshold
            elif node.condition == 'greater_than_or_equal':
                go_left = sample[node.feature] >= node.threshold
            else:
                raise ValueError(f"Unknown condition: {node.condition}")
            return predict_sample(node.children[0] if go_left else node.children[1], sample)

        return np.array([predict_sample(self.tree_, sample) for _, sample in X.iterrows()])

    def predict_proba(self, X):
        check_is_fitted(self)
        # Convert to DataFrame if it's a numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_)
        else:
            X = check_array(X, force_all_finite='allow-nan', ensure_2d=True, dtype=None)
            X = pd.DataFrame(X, columns=self.feature_names_)
        return self._predict_proba(X)

    def _predict_proba(self, X: pd.DataFrame):
        def predict_proba_sample(node: RuleNode, sample: pd.Series) -> np.ndarray:
            if node.feature is None:
                # This is a leaf node
                proba = np.array(node.class_counts)
                if proba.sum() == 0:
                    # If the node value is all zeros, return uniform distribution
                    return np.ones(self.n_classes_) / self.n_classes_
                return proba / proba.sum()

            if node.condition == 'less_than':
                go_left = sample[node.feature] < node.threshold
            elif node.condition == 'less_than_or_equal':
                go_left = sample[node.feature] <= node.threshold
            elif node.condition == 'greater_than':
                go_left = sample[node.feature] > node.threshold
            elif node.condition == 'greater_than_or_equal':
                go_left = sample[node.feature] >= node.threshold
            else:
                raise ValueError(f"Unknown condition: {node.condition}")

            if go_left:
                return predict_proba_sample(node.children[0], sample)
            else:
                return predict_proba_sample(node.children[1], sample)

        probas = []
        for _, sample in X.iterrows():
            proba = predict_proba_sample(self.tree_, sample)
            if len(proba) != self.n_classes_:
                # If the probability array doesn't match the number of classes,
                # pad it with zeros or truncate it
                proba = np.pad(proba, (0, max(0, self.n_classes_ - len(proba))), mode='constant')
                proba = proba[:self.n_classes_]
                proba /= proba.sum()  # Renormalize
            probas.append(proba)

        return np.array(probas)
    def score(self, X: pd.DataFrame, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)

    def enrich_custom_tree(self, X: pd.DataFrame, y: np.ndarray) -> RuleNode:
        if self.tree_ is None:
            return None

        unique_classes = np.unique(y)
        total_samples = len(y)
        class_counts = np.bincount(y)

        def enrich_node(node: RuleNode, node_X: pd.DataFrame, node_y: np.ndarray) -> None:
            node.samples = len(node_y)
            node.class_counts = np.bincount(node_y, minlength=len(unique_classes)).tolist()
            node.probas = (np.array(node.class_counts) / node.samples).tolist() if node.samples > 0 else np.zeros_like(
                node.class_counts, dtype=float).tolist()
            node.coverage = (np.array(node.class_counts) / class_counts).tolist()

            if node.feature is not None:
                if node.condition == 'less_than':
                    mask = node_X[node.feature] < node.threshold
                elif node.condition == 'less_than_or_equal':
                    mask = node_X[node.feature] <= node.threshold
                elif node.condition == 'greater_than':
                    mask = node_X[node.feature] > node.threshold
                elif node.condition == 'greater_than_or_equal':
                    mask = node_X[node.feature] >= node.threshold
                else:
                    raise ValueError(f"Unknown condition: {node.condition}")

                enrich_node(node.children[0], node_X[mask], node_y[mask])
                enrich_node(node.children[1], node_X[~mask], node_y[~mask])

        enrich_node(self.tree_, X, y)
        return self.tree_

    def _deep_copy_rules(self, rules: Dict[str, Any]) -> Dict[str, Any]:
        import copy
        return copy.deepcopy(rules)

    def get_enriched_rules(self) -> Dict[str, Any]:
        if self.enriched_rules is None:
            raise NotFittedError("The CustomDecisionTree has not been fitted yet. Call 'fit' before using this method.")
        return self.enriched_rules

    def get_custom_rules_model(self) -> Dict[str, Any]:
        if self.tree_ is None:
            raise NotFittedError("The CustomDecisionTree has not been fitted yet.")

        def node_to_dict(node: RuleNode) -> Dict[str, Any]:
            node_dict = {
                "name": node.name,
                "feature": node.feature,
                "condition": node.condition,
                "threshold": node.threshold,
                "class_counts": node.class_counts,
                "samples": node.samples,
                "targets": list(self.TargetMap.values()),
                "probas": node.probas,
                "coverage": node.coverage,
                "is_custom": node.is_custom,
                "ignore_after": node.ignore_after,
                "features_to_use_next": node.features_to_use_next
            }

            if node.feature is not None:
                node_dict["children"] = [node_to_dict(child) for child in node.children]

            return node_dict

        custom_rules_model = {
            "root": node_to_dict(self.tree_),
            "feature_names": self.feature_names_,
            "n_classes": self.n_classes_,
            "classes": self.classes_.tolist()
        }

        return custom_rules_model

    def load_from_sklearn_tree(self, sklearn_tree: DecisionTreeClassifier, X: pd.DataFrame, y: np.ndarray):
        self.feature_names_ = X.columns.tolist()
        self.classes_ = sklearn_tree.classes_
        self.n_classes_ = len(self.classes_)
        tree = sklearn_tree.tree_

        # Map class labels to indices for alignment
        class_indices = {cls: idx for idx, cls in enumerate(self.classes_)}

        # Get total class counts from y
        total_class_counts = np.zeros(len(self.classes_), dtype=float)
        for cls in y:
            idx = class_indices[cls]
            total_class_counts[idx] += 1

        # Get the decision path for each sample
        node_indicator = sklearn_tree.decision_path(X)

        # For each node, collect indices of samples that reach it
        node_sample_indices = {}
        for sample_id in range(X.shape[0]):
            node_indices = node_indicator.indices[node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]]
            for node_id in node_indices:
                if node_id not in node_sample_indices:
                    node_sample_indices[node_id] = []
                node_sample_indices[node_id].append(sample_id)

        def build_custom_tree(node_id: int, depth: int = 0) -> RuleNode:
            feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]

            # add model probabilities as well
            model_probas = tree.value[node_id][0]

            # Get samples that reach this node
            samples_at_node = node_sample_indices.get(node_id, [])
            n_node_samples = len(samples_at_node)

            # Get class counts at this node
            y_at_node = y[samples_at_node]
            class_counts = np.zeros(len(self.classes_), dtype=float)
            for cls in y_at_node:
                idx = class_indices[cls]
                class_counts[idx] += 1

            if feature != -2:  # Not a leaf node
                feature_name = self.feature_names_[feature]
                custom_node = RuleNode(
                    name=f"node_{node_id}",
                    feature=feature_name,
                    condition='less_than_or_equal',
                    threshold=threshold,
                    class_counts=class_counts.tolist(),
                    samples=n_node_samples,
                    is_custom=False
                )
                left_child = build_custom_tree(tree.children_left[node_id], depth + 1)
                right_child = build_custom_tree(tree.children_right[node_id], depth + 1)
                custom_node.add_child(left_child)
                custom_node.add_child(right_child)
            else:  # Leaf node
                custom_node = RuleNode(
                    name=f"leaf_{node_id}",
                    feature=None,
                    condition=None,
                    class_counts=class_counts.tolist(),
                    samples=n_node_samples,
                    is_custom=False
                )

            # Calculating probabilities safely
            probas_denominator = np.sum(class_counts)
            if probas_denominator > 0:
                custom_node.probas = (class_counts / probas_denominator).tolist()
            else:
                custom_node.probas = [0.0] * len(class_counts)

            # Calculating coverage per class
            coverage = np.zeros(len(self.classes_), dtype=float)
            for idx, total_count in enumerate(total_class_counts):
                if total_count > 0:
                    coverage[idx] = class_counts[idx] / total_count
                else:
                    coverage[idx] = 0.0
            custom_node.coverage = coverage.tolist()
            return custom_node

        self.tree_ = build_custom_tree(0)
        self.enriched_rules = self.tree_

        return self





    def generate_metrics(self, X: pd.DataFrame = None,
                         y: np.ndarray = None,
                         split_column: str = None,
                         num_splits: int = 10,
                         group_col: str = None) -> List:
        '''
        Generate metrics based on the splits in the split_column, or generate a Stratified split result if
        split_column is empty, if group_col is not None use StratifiedGroup folding.
        '''

        if X is None or y is None:
            raise ValueError("Both X and y must be provided.")

        performance_list = []

        # Get the custom decision tree
        custom_tree = self  # Assuming this method is part of the CustomDecisionTree class

        # Make the splitter
        if split_column:
            splits = X[split_column].unique()
            if len(splits) < num_splits:
                num_splits = len(splits)
        elif group_col:
            splitter = StratifiedGroupKFold(n_splits=num_splits, shuffle=True, random_state=42)
            splits = list(splitter.split(X, y, groups=X[group_col]))
        else:
            splitter = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
            splits = list(splitter.split(X, y))

        # Prepare for multi-class ROC AUC
        lb = LabelBinarizer()
        y_bin = lb.fit_transform(y)
        n_classes = y_bin.shape[1]

        # Go through the splits, perform train/test scoring with f1, recall, precision, roc_auc per class
        for i in range(num_splits):
            if split_column:
                train_idx = X[split_column] != splits[i]
                test_idx = X[split_column] == splits[i]
            else:
                train_idx, test_idx = splits[i]

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit the custom decision tree
            custom_tree.fit(X_train, y_train)

            # Make predictions
            y_pred = custom_tree.predict(X_test)
            y_pred_proba = custom_tree.predict_proba(X_test)

            # Calculate metrics
            metrics = {
                'split': i,
                'f1_micro': f1_score(y_test, y_pred, average='micro'),
                'f1_macro': f1_score(y_test, y_pred, average='macro'),
                'recall_micro': recall_score(y_test, y_pred, average='micro'),
                'recall_macro': recall_score(y_test, y_pred, average='macro'),
                'precision_micro': precision_score(y_test, y_pred, average='micro'),
                'precision_macro': precision_score(y_test, y_pred, average='macro'),
            }

            # Calculate ROC AUC for each class
            for j in range(n_classes):
                metrics[f'roc_auc_class_{j}'] = roc_auc_score(y_bin[test_idx][:, j], y_pred_proba[:, j])

            performance_list.append(metrics)

        return performance_list
def generate_sample_rules():
    """Generate a sample rule set for testing purposes."""
    rules = {
        "target_col": "Diagnosis",
        "fold_split_col": "Dataset",
        "features_to_use": [],
        "categorical_columns": [],
        "root": {
            "name": "root-node",
            "feature": "feature_0",
            "condition": "less_than",
            "value": 0,
            "children": [
                {
                    "name": "left-child",
                    "feature": "feature_1",
                    "condition": "higher_than",
                    "value": 0,
                    "pre_condition_value": True,
                    "children": []
                },
                {
                    "name": "right-child",
                    "feature": "feature_2",
                    "condition": "less_than",
                    "value": 0,
                    "pre_condition_value": False,
                    "children": []
                }
            ]
        }
    }
    return rules

def update_html(html_path: str = "./treeTemplate.html",
                tree: str = None,
                output_path: str = '../artifacts/decision_tree_visualization_with_data.html'):
    with open(html_path, 'r') as file:
        html_content = file.read()

    tree_json = json.dumps(tree)

    html_content = html_content.replace('<!-- Your JSON data will be inserted here -->',
                                        f'const treeData = {tree_json};')

    with open(output_path, 'w') as file:
        file.write(html_content)




if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Tree parser")
    argparser.add_argument("--rules_path", type=str, default=None, required=True)
    argparser.add_argument("--data_path", type=str, default=None, required=True)
    argparser.add_argument("--make_viz", type=bool, default=True, required=False)
    argparser.add_argument("--tree_config", type=str, default=None, required=True)
    parsed = argparser.parse_args()

    rules_path = parsed['rules_path']
    data_path = parsed['data_path']
    make_viz = parsed['make_viz']
    tree_config = parsed['tree_config']

    if tree_config is not None:
        # use benedict to open the YAML file
        TreeKwargs = benedict.benedict.from_yaml(tree_config)
    else:
        TreeKwargs = {'criterion': 'gini',
                      'max_depth': 5,
                      'random_state': 42}

    if data_path is not None:
        # must be parquet
        assert(data_path.endswith('.parquet')), 'File must be a parquet, sorry not sorry'
        # Load the dataset
        df = pd.read_parquet(data_path)

    if rules_path is not None:
        print(f"Processing rules from {rules_path}")
        print(f"Data from {data_path}")

        rules_loader = LoadRules(rules_path)
        processed_rules = rules_loader.get_processed_rules()

        clf = CustomDecisionTree(custom_rules=processed_rules, Tree_kwargs=TreeKwargs)
        clf.fit(X_train, y_train)
        # Make predictions
        y_pred = clf.predict(X_test)
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.4f}")

        # Get and print enriched rules
        enriched_rules = clf.get_enriched_rules()
        print("\nEnriched Rules:")
        print(json.dumps(enriched_rules, indent=2))
        with open("temp_rules_enriched_v2.json", "w") as f:
            json.dump(enriched_rules, f)

        final_tree = clf.get_custom_rules_model(X_train, y_train)
        print("\nFinal Custom Decision Tree:")
        print(json.dumps(final_tree, indent=3))
        print(30 * "-")
        with open("temp_rules_final_tree_v2.json", "w") as f:
            json.dump(final_tree, f)
        print("\nWriting html for tree:")
        update_html(tree=final_tree)
    else:
        # Assume basic test
        # Generate a sample dataset
        X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=2,
                                   n_classes=2, random_state=42)

        # Convert to DataFrame for named features
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Generate sample rules and save to a temporary file
        rules = generate_sample_rules()
        with open("temp_rules.json", "w") as f:
            json.dump(rules, f)

        # Load the rules
        rules_loader = LoadRules("temp_rules.json")
        processed_rules = rules_loader.get_processed_rules()

        # Create and train the custom decision tree
        clf = CustomDecisionTree(custom_rules=processed_rules, criterion='gini', max_depth=5, random_state=42)
        clf.fit(X_train, y_train)

        # Make predictions
        y_pred = clf.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.4f}")

        # Get and print enriched rules
        enriched_rules = clf.get_enriched_rules()
        print("\nEnriched Rules:")
        print(json.dumps(enriched_rules, indent=2))
        with open("temp_rules_enriched_v2.json", "w") as f:
            json.dump(enriched_rules, f)

        final_tree = clf.get_custom_rules_model(X_train, y_train)
        print("\nFinal Custom Decision Tree:")
        print(json.dumps(final_tree, indent=3))
        print(30 * "-")
        with open("temp_rules_final_tree_v2.json", "w") as f:
            json.dump(final_tree, f)

        # Compare with standard DecisionTreeClassifier
        std_clf = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
        std_clf.fit(X_train, y_train)
        std_y_pred = std_clf.predict(X_test)
        std_accuracy = accuracy_score(y_test, std_y_pred)
        print(f"\nStandard DecisionTreeClassifier accuracy: {std_accuracy:.4f}")

        print("\nWriting html for tree:")
        update_html(tree=final_tree)
