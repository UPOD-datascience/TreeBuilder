# TreeBuilder

This is a tool to build semi-custom trees. Starting a from a custom tree, the tree is continued using sklearn's DecisionTreeClassifier. The tree is then pruned using a custom pruning algorithm. The tree is then converted to a semi-custom tree by replacing the leaf nodes with a custom model. The resulting tree can be written to an interactive HTML file.

<div align="center">
    <img src="images/TreeBuilder.webp" alt="TreeBuilder">
</div>

Basic usage:


```python

from TreeBuilder import CustomDecisionTree, LoadRules, update_html
rules_loader = LoadRules(path_to_rules_json)
processed_rules = rules_loader.get_processed_rules()
treeBuilder = CustomDecisionTree(processed_rules)

# Train the tree
treeBuilder.fit(X_train, y_train)
y_test = treeBuilder.predict(X_test)

# Visualize the tree
enriched_rules = treeBuilder.get_enriched_rules()
final_tree = treeBuilder.get_custom_rules_model()
update_html(final_tree, output_path='bla')
```

A custom rules JSON might look like this
```json
{
  "fold_split_col": "Dataset",
  "target_col": "TARGET",
  "ignore_cols": ["X9"],
  "features_to_use": [],
  "threshold": 0.9,
  "root": {
    "name": "root-node",
    "feature": "X4",
    "condition": "higher_than_or_equal_to",
    "value": 50,                                    
        "children": [
          {
            "name": "root-child-1",
            "pre_condition_value": true,
            "feature": "X1",
            "condition": "higher_than_or_equal_to",
            "value": 100,
            "ignore_after": ["X2",
                             "X3"]
          },
          {
            "name": "root-child-2",
            "pre_condition_value": false,
            "feature": null,
            "condition": null,
            "value":null ,
            "features_to_use_next":  ["X2",
                                      "X3",
                                      "X4"]
         }
        ]

    }
}      
```

This can be as specific as you like.

You can also write out a normal Sklearn Decision tree to an HTML file using the following code:

```python
treeBuilder = CustomDecisionTree()
sklearn_tree = treeBuilder.load_from_sklearn_tree(clf_base, X_train_imputed, y_train_encoded)
final_tree_sklearn = sklearn_tree.get_custom_rules_model()
update_html(final_tree_sklearn, output_path='bla')
```
