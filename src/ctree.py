import numpy as np
import pandas as pd
import json
from typing import Dict, Any, List, Tuple, Union

from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, pre_condition_value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.pre_condition_value = pre_condition_value

class LoadRules:
    def __init__(self, rules_file: str):
        self.rules = self._load_rules(rules_file)

    def _load_rules(self, rules_file: str) -> Dict[str, Any]:
        """
        Load custom rules from a JSON file.

        :param rules_file: Path to the JSON file containing the rules
        :return: Loaded rules as a dictionary
        """
        with open(rules_file, 'r') as f:
            rules = json.load(f)
        return rules

    def _process_rules(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively process the rules from the JSON structure.

        :param node: Current node in the JSON structure
        :return: Processed node
        """
        processed_node = {
            "name": node["name"],
            "feature": node["feature"],
            "condition": node["condition"],
            "value": node["value"],
        }

        if "pre_condition_value" in node:
            processed_node["pre_condition_value"] = node["pre_condition_value"]

        if "children" in node:
            processed_node["children"] = [self._process_rules(child) for child in node["children"]]

        return processed_node

    def get_processed_rules(self) -> Dict[str, Any]:
        """
        Get the processed rules.

        :return: Processed rules as a dictionary
        """
        return self._process_rules(self.rules["root"])


class CustomDecisionTreeV1(BaseEstimator, ClassifierMixin):
    def __init__(self, custom_rules: Dict[str, Any] = None, criterion='gini', max_depth=None, random_state=None,
                 prune_threshold=None):
        '''
        Custom decision tree builder v1; this applies the custom tree first and then continues
        building a tree on the remaining unmasked data.

        :param custom_rules: Dictionary of custom rules
        :param criterion: the decision tree criterion
        :param max_depth: maximum depth of the decision tree
        :param random_state: the seed
        :param prune_threshold: what is the prune probability threshold
        '''
        self.custom_rules = custom_rules
        self.criterion = criterion
        self.max_depth = max_depth
        self.random_state = random_state
        self.prune_threshold = prune_threshold
        self.tree_ = None
        self.enriched_rules = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        # Apply custom rules to get the mask of valid samples
        mask = self.apply_custom_rules(X)

        # Fit a decision tree on the remaining data after custom splits
        self.tree_ = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth,
                                            random_state=self.random_state)
        self.tree_.fit(X[mask], y[mask])

        # Apply pruning if threshold is set
        if self.prune_threshold is not None:
            self._apply_pruning(X[mask], y[mask])

        # Enrich custom tree with probabilities and coverage
        self.enriched_rules = self.enrich_custom_tree(X, y)

        return self

    def apply_custom_rules(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply custom rules to the input data.

        :param X: Input features
        :return: Boolean mask of samples that pass all custom rules
        """
        if self.custom_rules is None:
            return np.ones(X.shape[0], dtype=bool)

        return self._apply_node_rules(self.custom_rules, X)

    def _apply_node_rules(self, node: Dict[str, Any], X: pd.DataFrame) -> np.ndarray:
        """
        Recursively apply rules for a node and its children.

        :param node: Current node in the rule tree
        :param X: Input features
        :return: Boolean mask of samples that pass the rules for this node and its children
        """
        feature = node['feature']
        condition = node['condition']
        value = node['value']

        if condition == 'less_than':
            mask = X[feature] <= value
        elif condition == 'higher_than':
            mask = X[feature] > value
        else:
            raise ValueError(f"Unknown condition: {condition}")

        if 'pre_condition_value' in node:
            mask = mask if node['pre_condition_value'] else ~mask

        for child in node.get('children', []):
            child_mask = self._apply_node_rules(child, X)
            mask &= child_mask

        return mask

    def _apply_pruning(self, X: pd.DataFrame, y: np.ndarray):
        """ Apply pruning to the decision tree based on the prune_threshold. """
        node_indices = self.tree_.apply(X)
        node_values = self.tree_.tree_.value

        for node in np.unique(node_indices):
            node_samples = np.where(node_indices == node)[0]
            class_counts = np.bincount(y[node_samples])
            class_probabilities = class_counts / np.sum(class_counts)
            max_prob = np.max(class_probabilities)

            if max_prob >= self.prune_threshold:
                # Make this node a leaf by setting its children to -1
                self.tree_.tree_.children_left[node] = -1
                self.tree_.tree_.children_right[node] = -1

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.tree_ is None:
            raise NotFittedError("This CustomDecisionTree instance is not fitted yet")

        # Apply custom rules
        mask = self.apply_custom_rules(X)
        predictions = np.full(X.shape[0], -1)

        # Predict using the fitted tree for valid samples
        predictions[mask] = self.tree_.predict(X[mask])
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.tree_ is None:
            raise NotFittedError("This CustomDecisionTree instance is not fitted yet")

        # Apply custom rules
        mask = self.apply_custom_rules(X)
        probabilities = np.zeros((X.shape[0], len(self.tree_.classes_)))

        # Predict probabilities using the fitted tree for valid samples
        probabilities[mask] = self.tree_.predict_proba(X[mask])
        return probabilities

    def score(self, X: pd.DataFrame, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)

    def enrich_custom_tree(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """
        Enrich the custom tree dictionary with probabilities and coverage information.

        :param X: Input features
        :param y: Target values
        :return: Enriched custom tree dictionary
        """
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
                mask = X[feature] <= value
            elif condition == 'higher_than':
                mask = X[feature] > value
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
        """
        Create a deep copy of the custom rules dictionary.

        :param rules: Original custom rules dictionary
        :return: Deep copy of the custom rules dictionary
        """
        import copy
        return copy.deepcopy(rules)

    def get_enriched_rules(self) -> Dict[str, Any]:
        """
        Get the enriched custom tree dictionary.

        :return: Enriched custom tree dictionary
        """
        if self.enriched_rules is None:
            raise NotFittedError("The CustomDecisionTree has not been fitted yet. Call 'fit' before using this method.")
        return self.enriched_rules


class CustomDecisionTreeV2(BaseEstimator, ClassifierMixin):
    def __init__(self, custom_rules: Dict[str, Any] = None,
                 criterion='gini', max_depth=None, random_state=None, prune_threshold=0.9):
        '''
        Custom decision tree builder v2; this applies the custom tree first and then continues
        building trees from the leaves of the custom tree.

        :param custom_rules: Dictionary of custom rules
        :param criterion: the decision tree criterion
        :param max_depth: maximum depth of the decision tree
        :param random_state: the seed
        '''
        self.custom_rules = custom_rules
        self.criterion = criterion
        self.max_depth = max_depth
        self.random_state = random_state
        self.prune_threshold = prune_threshold
        self.tree_ = None
        self.feature_names_ = None
        self.enriched_rules = None
        self.features_to_consider = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.feature_names_ = X.columns.tolist()
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Extract features to consider for continuation
        self.features_to_consider = self._get_features_to_consider()

        # Create the initial tree structure based on custom rules
        self.tree_ = self._build_custom_tree(self.custom_rules, X, y)

        # Continue building the tree with DecisionTreeClassifier at the leaves
        self._continue_tree_building(X, y)

        # Prune the tree if a threshold is set
        if self.prune_threshold is not None:
            self._prune_tree(self.tree_)

        # Enrich custom tree with probabilities and coverage
        self.enriched_rules = self.enrich_custom_tree(X, y)

        return self
    def _get_features_to_consider(self) -> List[str]:
        if self.custom_rules and 'ignore_after' in self.custom_rules:
            ignore_features = set(self.custom_rules['ignore_after'])
            return [f for f in self.feature_names_ if f not in ignore_features]
        return self.feature_names_
    def _build_custom_tree(self, rules: Dict[str, Any], X: pd.DataFrame, y: np.ndarray) -> Node:
        return self._build_node(rules, X, y)

    def _build_node(self, node: Dict[str, Any], X: pd.DataFrame, y: np.ndarray) -> Node:
        if 'feature' not in node:
            # This is a leaf node
            return Node(value=np.bincount(y, minlength=self.n_classes_))

        feature = self.feature_names_.index(node['feature'])
        threshold = node['value']

        # Split the data
        if node['condition'] == 'less_than':
            left_mask = X.iloc[:, feature] <= threshold
        else:  # 'higher_than'
            left_mask = X.iloc[:, feature] > threshold

        left_X, left_y = X[left_mask], y[left_mask.values]
        right_X, right_y = X[~left_mask], y[~left_mask.values]

        # Create the current node
        current_node = Node(feature=feature, threshold=threshold)

        # Process children if they exist
        if 'children' in node and node['children']:
            for child in node['children']:
                pre_condition_value = child.get('pre_condition_value', True)
                if pre_condition_value:
                    current_node.left = self._build_node(child, left_X, left_y)
                    current_node.left.pre_condition_value = pre_condition_value
                else:
                    current_node.right = self._build_node(child, right_X, right_y)
                    current_node.right.pre_condition_value = pre_condition_value

        # If a child is not set, create a leaf node
        if current_node.left is None:
            current_node.left = Node(value=np.bincount(left_y, minlength=self.n_classes_))
        if current_node.right is None:
            current_node.right = Node(value=np.bincount(right_y, minlength=self.n_classes_))

        return current_node

    def _continue_tree_building(self, X: pd.DataFrame, y: np.ndarray):
        def build_subtree(node: Node, node_X: pd.DataFrame, node_y: np.ndarray):
            if node.feature is None:
                # This is a leaf in the custom tree, continue with DecisionTreeClassifier
                features_to_use = [f for f in self.features_to_consider if f in node_X.columns]
                subtree = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth,
                                                 random_state=self.random_state)
                subtree.fit(node_X[features_to_use], node_y)

                # Replace the leaf with the subtree
                if subtree.tree_.feature[0] != -2:  # -2 indicates a leaf in scikit-learn's implementation
                    node.feature = self.feature_names_.index(features_to_use[subtree.tree_.feature[0]])
                    node.threshold = subtree.tree_.threshold[0]
                    node.left = Node(value=subtree.tree_.value[subtree.tree_.children_left[0]][0])
                    node.right = Node(value=subtree.tree_.value[subtree.tree_.children_right[0]][0])

                    left_mask = node_X.iloc[:, node.feature] <= node.threshold
                    build_subtree(node.left, node_X[left_mask], node_y[left_mask.values])
                    build_subtree(node.right, node_X[~left_mask], node_y[~left_mask.values])
            else:
                # This is an internal node, process its children
                left_mask = node_X.iloc[:, node.feature] <= node.threshold
                build_subtree(node.left, node_X[left_mask], node_y[left_mask.values])
                build_subtree(node.right, node_X[~left_mask], node_y[~left_mask.values])

        # Start building subtrees from the root
        build_subtree(self.tree_, X, y)

    def _prune_tree(self, node: Node):
        if node.feature is None:  # Leaf node
            return

        # Recursively prune children
        self._prune_tree(node.left)
        self._prune_tree(node.right)

        # Check if this node should be pruned
        if node.left.feature is None and node.right.feature is None:
            left_prob = np.max(node.left.value / np.sum(node.left.value))
            right_prob = np.max(node.right.value / np.sum(node.right.value))

            if left_prob >= self.prune_threshold and right_prob >= self.prune_threshold:
                # Prune this node (make it a leaf)
                node.feature = None
                node.threshold = None
                node.value = node.left.value + node.right.value
                node.left = None
                node.right = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.tree_ is None:
            raise NotFittedError("This CustomDecisionTree instance is not fitted yet")

        def predict_sample(node: Node, sample: pd.Series) -> int:
            if node.feature is None:
                return np.argmax(node.value)
            if sample[self.feature_names_[node.feature]] <= node.threshold:
                return predict_sample(node.left, sample)
            else:
                return predict_sample(node.right, sample)

        return np.array([predict_sample(self.tree_, sample) for _, sample in X.iterrows()])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.tree_ is None:
            raise NotFittedError("This CustomDecisionTree instance is not fitted yet")

        def predict_proba_sample(node: Node, sample: pd.Series) -> np.ndarray:
            if node.feature is None:
                return node.value / np.sum(node.value)
            if sample[self.feature_names_[node.feature]] <= node.threshold:
                return predict_proba_sample(node.left, sample)
            else:
                return predict_proba_sample(node.right, sample)

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
                mask = X[feature] <= value
            elif condition == 'higher_than':
                mask = X[feature] > value
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

        def extract_node(node: Node, node_X: pd.DataFrame, node_y: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
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
                node_mask = node_X.iloc[:, node.feature] <= node.threshold
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
                    "condition": "less_than",
                    "value": numpy_to_python(node.threshold),
                    "samples": numpy_to_python(len(node_y)),
                    "probas": numpy_to_python(node_probas.tolist()),
                    "coverage": numpy_to_python(node_coverage.tolist()),
                    "children": [left_result, right_result]
                }

                return result, node_mask

        final_tree, _ = extract_node(self.tree_, X, y)
        return {"root": final_tree}


def generate_sample_rules():
    """Generate a sample rule set for testing purposes."""
    rules = {
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
    from sklearn.tree import DecisionTreeClassifier

    std_clf = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
    std_clf.fit(X_train, y_train)
    std_y_pred = std_clf.predict(X_test)
    std_accuracy = accuracy_score(y_test, std_y_pred)
    print(f"\nStandard DecisionTreeClassifier accuracy: {std_accuracy:.4f}")

    print("\nWriting html for tree:")
    update_html(tree=final_tree)
