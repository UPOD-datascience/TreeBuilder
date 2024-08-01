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
import dotenv
#TODO nice: add argparse to parse other .json's
#TODO nice: add feature recombinator
#TODO nice: option for a GradientBoostedClassifier followed by RuleFit

#[x] TODO must: add higher_than_or_equal_to
#TODO must: if root is empty, then create a decisiontree directly
#TODO must: add YAML file with model settings
#TODO must: add pickle function
#TODO must: add error handling in main functions

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
                 value: Optional[Any],
                 ignore_after: Optional[List[str]] = None,
                 features_to_use_next: Optional[List[str]] = None,
                 pre_condition_value: Optional[Any] = None):
        self.name = self._validate_non_empty_string(name, "name")
        self.feature = self._validate_feature(feature)
        self.condition = self._validate_condition(condition)
        self.value = value  # No validation for value as it can be None or any type
        self.ignore_after = ignore_after or []
        self.features_to_use_next = features_to_use_next or []
        self.pre_condition_value = pre_condition_value
        self.children: List[RuleNode] = []

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

    def add_child(self, child: 'RuleNode') -> None:
        self.children.append(child)

class LoadRules:
    def __init__(self, rules_file: str):
        self.rules = self._load_rules(rules_file)
        self.fold_split_col = self.rules.get("fold_split_col")
        self.target_col = self.rules.get("target_col")
        self.ignore_cols = self.rules.get("ignore_cols", [])
        self.features_to_use = self.rules.get("features_to_use", [])
        self.root = self._process_rules(self.rules["root"])

    def _load_rules(self, rules_file: str) -> Dict[str, Any]:
        with open(rules_file, 'r') as f:
            rules = json.load(f)
        return rules

    def _process_rules(self, node: Dict[str, Any]) -> RuleNode:
        processed_node = RuleNode(
            name=node["name"],
            feature=node.get("feature"),
            condition=node.get("condition"),
            value=node.get("value"),
            ignore_after=node.get("ignore_after"),
            features_to_use_next=node.get("features_to_use_next")
        )

        if "pre_condition_value" in node:
            processed_node.pre_condition_value = node["pre_condition_value"]

        if "children" in node:
            for child in node["children"]:
                processed_node.add_child(self._process_rules(child))

        return processed_node

    def get_processed_rules(self) -> RuleNode:
        return self.root

class CustomDecisionTreeV2(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 custom_rules: Union[Dict[str, Any], RuleNode] = None,
                 criterion: str='gini', max_depth: int=None, random_state: int=None,
                 prune_threshold: float=0.9,
                 Tree_kwargs: Dict[str, Any] = None):
        '''
        Custom decision tree builder v2; this applies the custom tree first and then continues
        building trees from the leaves of the custom tree.

        :param custom_rules: Dictionary of custom rules
        :param criterion: the decision tree criterion
        :param max_depth: maximum depth of the decision tree
        :param random_state: the seed
        '''
        self.custom_rules = custom_rules
        self.prune_threshold = prune_threshold
        self.tree_ = None
        self.feature_names_ = None
        self.enriched_rules = None
        self.features_to_consider = None
        self.Tree_kwargs = Tree_kwargs or {}
        if Tree_kwargs is None:
            self.Tree_kwargs['criterion'] = criterion
            self.Tree_kwargs['max_depth'] = max_depth
            self.Tree_kwargs['random_state'] = random_state

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.feature_names_ = X.columns.tolist()
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Extract features to consider for continuation
        self.features_to_consider = self._get_features_to_consider(X)

        # Create the initial tree structure based on custom rules
        if isinstance(self.custom_rules, dict):
            self.tree_ = self._build_custom_tree(self.custom_rules['root'], X, y)
        elif isinstance(self.custom_rules, RuleNode):
            self.tree_ = self.custom_rules
        else:
            raise ValueError("custom_rules must be either a dictionary or a RuleNode object")

        # Continue building the tree with DecisionTreeClassifier at the leaves
        self._continue_tree_building(X, y)

        # Prune the tree if a threshold is set
        if self.prune_threshold is not None:
            self._prune_tree(self.tree_)

        # Enrich custom tree with probabilities and coverage
        self.enriched_rules = self.enrich_custom_tree(X, y)

        return self
    def _get_matching_features(self, feature_list: List[str], all_features: List[str]) -> List[str]:
        """
        Find all features in all_features that contain any of the strings in feature_list.
        """
        return [f for f in all_features if any(spec in f for spec in feature_list)]

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

    def _build_custom_tree(self, node: Dict[str, Any], X: pd.DataFrame, y: np.ndarray) -> RuleNode:
        if 'feature' not in node:
            # This is a leaf node
            return RuleNode(name="leaf", feature=None, condition=None, value=np.bincount(y, minlength=self.n_classes_).tolist())

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
        current_node = RuleNode(name=node['name'], feature=feature, condition=condition, value=threshold,
                                ignore_after=ignore_after, features_to_use_next=features_to_use_next)

        # Process children if they exist
        if 'children' in node and node['children']:
            for child in node['children']:
                pre_condition_value = child.get('pre_condition_value', True)
                child_node = self._build_custom_tree(child, left_X if pre_condition_value else right_X,
                                                     left_y if pre_condition_value else right_y)
                current_node.add_child(child_node)

        # If a child is not set, create a leaf node
        if len(current_node.children) == 0:
            current_node.add_child(RuleNode(name="left_leaf", feature=None, condition=None,
                                            value=np.bincount(left_y, minlength=self.n_classes_).tolist()))
            current_node.add_child(RuleNode(name="right_leaf", feature=None, condition=None,
                                            value=np.bincount(right_y, minlength=self.n_classes_).tolist()))

        return current_node

    def _continue_tree_building(self, X: pd.DataFrame, y: np.ndarray):
        def build_subtree(node: RuleNode, node_X: pd.DataFrame, node_y: np.ndarray):
            if node.feature is None:
                # This is a leaf in the custom tree, continue with DecisionTreeClassifier
                features_to_use = self._get_matching_features(node.features_to_use_next, node_X.columns) if node.features_to_use_next else [f for f in self.features_to_consider if f not in self._get_matching_features(node.ignore_after, node_X.columns)]
                subtree = DecisionTreeClassifier(**self.Tree_kwargs)
                subtree.fit(node_X[features_to_use], node_y)

                # Replace the leaf with the subtree
                if subtree.tree_.feature[0] != -2:  # -2 indicates a leaf in scikit-learn's implementation
                    node.feature = features_to_use[subtree.tree_.feature[0]]
                    node.condition = 'less_than_or_equal'
                    node.value = subtree.tree_.threshold[0]
                    node.children = [
                        RuleNode(name="left_leaf", feature=None, condition=None, value=subtree.tree_.value[subtree.tree_.children_left[0]][0].tolist()),
                        RuleNode(name="right_leaf", feature=None, condition=None, value=subtree.tree_.value[subtree.tree_.children_right[0]][0].tolist())
                    ]

                    left_mask = node_X[node.feature] <= node.value
                    build_subtree(node.children[0], node_X[left_mask], node_y[left_mask])
                    build_subtree(node.children[1], node_X[~left_mask], node_y[~left_mask])
            else:
                # This is an internal node, process its children
                for child in node.children:
                    if node.condition == 'less_than' or node.condition == 'less_than_or_equal':
                        mask = node_X[node.feature] <= node.value
                    else:
                        mask = node_X[node.feature] > node.value
                    build_subtree(child, node_X[mask], node_y[mask])

        # Start building subtrees from the root
        build_subtree(self.tree_, X, y)

    def _prune_tree(self, node: RuleNode):
        if node.feature is None:  # Leaf node
            return

        # Recursively prune children
        for child in node.children:
            self._prune_tree(child)

        # Check if this node should be pruned
        if all(child.feature is None for child in node.children):
            left_prob = np.max(node.children[0].value) / np.sum(node.children[0].value)
            right_prob = np.max(node.children[1].value) / np.sum(node.children[1].value)

            if left_prob >= self.prune_threshold and right_prob >= self.prune_threshold:
                # Prune this node (make it a leaf)
                node.feature = None
                node.condition = None
                node.value = [sum(x) for x in zip(node.children[0].value, node.children[1].value)]
                node.children = []

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.tree_ is None:
            raise NotFittedError("This CustomDecisionTree instance is not fitted yet")

        def predict_sample(node: RuleNode, sample: pd.Series) -> int:
            if node.feature is None:
                return np.argmax(node.value)
            if node.condition in ['less_than', 'less_than_or_equal']:
                go_left = sample[node.feature] <= node.value
            else:
                go_left = sample[node.feature] > node.value
            return predict_sample(node.children[0] if go_left else node.children[1], sample)

        return np.array([predict_sample(self.tree_, sample) for _, sample in X.iterrows()])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.tree_ is None:
            raise NotFittedError("This CustomDecisionTree instance is not fitted yet")

        def predict_proba_sample(node: RuleNode, sample: pd.Series) -> np.ndarray:
            if node.feature is None:
                return np.array(node.value) / np.sum(node.value)
            if node.condition in ['less_than', 'less_than_or_equal']:
                go_left = sample[node.feature] <= node.value
            else:
                go_left = sample[node.feature] > node.value
            return predict_proba_sample(node.children[0] if go_left else node.children[1], sample)

        return np.array([predict_proba_sample(self.tree_, sample) for _, sample in X.iterrows()])

    def score(self, X: pd.DataFrame, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)

    def enrich_custom_tree(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        if self.custom_rules is None:
            return None

        enriched_rules = self._deep_copy_rules(self.custom_rules)
        unique_classes = np.unique(y)
        total_samples = len(y)
        class_counts = np.bincount(y)

        def enrich_node(node: Dict[str, Any], parent_mask: np.ndarray) -> np.ndarray:
            feature = node['feature']
            condition = node['condition']
            value = node['value']

            if condition == 'less_than':
                mask = X[feature] < value
            elif condition == 'less_than_or_equal':
                mask = X[feature] <= value
            elif condition == 'greater_than':
                mask = X[feature] > value
            elif condition == 'greater_than_or_equal':
                mask = X[feature] >= value
            else:
                raise ValueError(f"Unknown condition: {condition}")

            if 'pre_condition_value' in node:
                mask = mask if node['pre_condition_value'] else ~mask

            mask &= parent_mask

            node_y = y[mask]
            node_counts = np.bincount(node_y, minlength=len(unique_classes))
            node_probas = node_counts / len(node_y) if len(node_y) > 0 else np.zeros_like(node_counts, dtype=float)
            node_coverage = node_counts / class_counts

            node['probas'] = node_probas.tolist()
            node['coverage'] = node_coverage.tolist()

            for child in node.get('children', []):
                enrich_node(child, mask)

            return mask

        enrich_node(enriched_rules, np.ones(X.shape[0], dtype=bool))
        return enriched_rules

    def _deep_copy_rules(self, rules: Dict[str, Any]) -> Dict[str, Any]:
        import copy
        return copy.deepcopy(rules)

    def get_enriched_rules(self) -> Dict[str, Any]:
        if self.enriched_rules is None:
            raise NotFittedError("The CustomDecisionTree has not been fitted yet. Call 'fit' before using this method.")
        return self.enriched_rules

    def get_custom_rules_model(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        if self.tree_ is None:
            raise NotFittedError("The CustomDecisionTree has not been fitted yet.")

        def numpy_to_python(obj: Any) -> Union[int, float, list, Dict[str, Any]]:
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return [numpy_to_python(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: numpy_to_python(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [numpy_to_python(item) for item in obj]
            else:
                return obj

        def extract_node(node: RuleNode, node_X: pd.DataFrame, node_y: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
            if node.feature is None:
                node_mask = np.ones(len(node_y), dtype=bool)
                node_counts = np.bincount(node_y, minlength=self.n_classes_)
                node_probas = node_counts / len(node_y) if len(node_y) > 0 else np.zeros_like(node_counts, dtype=float)
                node_coverage = node_counts / np.bincount(y, minlength=self.n_classes_)

                return {
                    "name": "leaf",
                    "value": numpy_to_python(node.value.tolist() if node.value is not None else None),
                    "samples": numpy_to_python(np.sum(node.value) if node.value is not None else 0),
                    "probas": numpy_to_python(node_probas.tolist()),
                    "coverage": numpy_to_python(node_coverage.tolist())
                }, node_mask
            else:
                if node.condition == 'less_than':
                    node_mask = node_X.iloc[:, node.feature] < node.threshold
                elif node.condition == 'less_than_or_equal':
                    node_mask = node_X.iloc[:, node.feature] <= node.threshold
                elif node.condition == 'higher_than':
                    node_mask = node_X.iloc[:, node.feature] > node.threshold
                elif node.condition == 'higher_than_or_equal':
                    node_mask = node_X.iloc[:, node.feature] >= node.threshold
                else:
                    raise ValueError(f"Unknown condition: {node.condition}")

                left_X, left_y = node_X[node_mask], node_y[node_mask]
                right_X, right_y = node_X[~node_mask], node_y[~node_mask]

                left_result, left_mask = extract_node(node.left, left_X, left_y)
                right_result, right_mask = extract_node(node.right, right_X, right_y)

                node_counts = np.bincount(node_y, minlength=self.n_classes_)
                node_probas = node_counts / len(node_y) if len(node_y) > 0 else np.zeros_like(node_counts, dtype=float)
                node_coverage = node_counts / np.bincount(y, minlength=self.n_classes_)

                result = {
                    "name": f"node-{self.feature_names_[node.feature]}",
                    "feature": self.feature_names_[node.feature],
                    "condition": node.condition,  # Use the node's condition
                    "value": numpy_to_python(node.threshold),
                    "samples": numpy_to_python(len(node_y)),
                    "probas": numpy_to_python(node_probas.tolist()),
                    "coverage": numpy_to_python(node_coverage.tolist()),
                    "children": [left_result, right_result]
                }

                return result, node_mask

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

def update_html(html_path: str="./treeTemplate.html",
                tree: str=None):
    with open(html_path, 'r') as file:
        html_content = file.read()

    tree_json = json.dumps(tree)

    html_content = html_content.replace('<!-- Your JSON data will be inserted here -->',
                                        f'const treeData = {tree_json};')

    with open('../artifacts/decision_tree_visualization_with_data.html', 'w') as file:
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

        clf = CustomDecisionTreeV2(custom_rules=processed_rules, Tree_kwargs=TreeKwargs)
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
        print("CustomDecisionTreeV1")
        print(30 * "+")
        clf = CustomDecisionTreeV1(custom_rules=processed_rules, criterion='gini', max_depth=5, random_state=42,
                                 prune_threshold=0.95)
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

        with open("temp_rules_enriched_v1.json", "w") as f:
            json.dump(enriched_rules, f)

        print(30*"+")
        print("CustomDecisionTreeV2")
        print(30 * "+")

        clf = CustomDecisionTreeV2(custom_rules=processed_rules, criterion='gini', max_depth=5, random_state=42)
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
