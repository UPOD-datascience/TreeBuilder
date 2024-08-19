import numpy as np
from scipy import stats
import pandas as pd
import pickle
import os
import tqdm

from typing import List, Callable, Dict, Tuple

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn.calibration import calibration_curve
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted

#TODO: add rulefitter
#TODO: add l1 regularised logistic regression

def rulefitter():
    # RuleFit: https://github.com/christophM/rulefit
    # Imodels: https://github.com/csinva/imodels
    # Skope-rules: https://github.com/scikit-learn-contrib/skope-rules
    pass


class GenericCalibratedClassifier(BaseEstimator, ClassifierMixin):
    """
    A generic wrapper for classifiers that adds naive probability calibration.
    This wrapper scales the output of the classifier's decision function or
    probability estimates to [0, 1] for each class.
    """

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.scaler = None

    def fit(self, X, y, sample_weight=None):
        # Fit the base estimator
        self.base_estimator.fit(X, y, sample_weight=sample_weight)

        # Get the raw predictions (decision function or probabilities)
        if hasattr(self.base_estimator, "decision_function"):
            raw_predictions = self.base_estimator.decision_function(X)
        elif hasattr(self.base_estimator, "predict_proba"):
            raw_predictions = self.base_estimator.predict_proba(X)
        else:
            raise AttributeError("Base estimator must have either 'decision_function' or 'predict_proba' method.")

        # Reshape if necessary (for binary classification)
        if raw_predictions.ndim == 1:
            raw_predictions = raw_predictions.reshape(-1, 1)

        # Fit a MinMaxScaler for each class
        self.scaler = MinMaxScaler()
        self.scaler.fit(raw_predictions)

        return self

    def predict_proba(self, X):
        check_is_fitted(self, ['scaler'])

        # Get the raw predictions
        if hasattr(self.base_estimator, "decision_function"):
            raw_predictions = self.base_estimator.decision_function(X)
        elif hasattr(self.base_estimator, "predict_proba"):
            raw_predictions = self.base_estimator.predict_proba(X)
        else:
            raise AttributeError("Base estimator must have either 'decision_function' or 'predict_proba' method.")

        # Reshape if necessary (for binary classification)
        if raw_predictions.ndim == 1:
            raw_predictions = raw_predictions.reshape(-1, 1)

        # Scale the predictions
        scaled_predictions = self.scaler.transform(raw_predictions)

        # Clip values to ensure they're in the [0, 1] range
        scaled_predictions = np.clip(scaled_predictions, 0, 1)

        # If binary classification, return probabilities for both classes
        if scaled_predictions.shape[1] == 1:
            return np.hstack([1 - scaled_predictions, scaled_predictions])

        # For multiclass, ensure probabilities sum to 1
        row_sums = scaled_predictions.sum(axis=1)
        return scaled_predictions / row_sums[:, np.newaxis]

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.base_estimator.classes_[np.argmax(probas, axis=1)]


def net_benefit_curve_plot(results: pd.DataFrame = None,
                           true_col_prefix='Y_test',
                           pred_col_prefix='Y_pred',
                           output_path="",
                           threshold_steps=20,
                           xlim=[0, 0.5],
                           ylim=[-0.1, 0.1],
                           show_plot=False,
                           plot_title=""):
    """
    Plot net benefit curves for each target, with separate plots per target.

    Args:
    results (pd.DataFrame): DataFrame containing true values and predictions.
    true_col_prefix (str): Prefix for columns containing true values.
    pred_col_prefix (str): Prefix for columns containing predictions.
    output_path (str): Path to save the output plots.
    threshold_steps (int): Number of threshold steps to evaluate.
    xlim (list): X-axis limits for the plot.
    ylim (list): Y-axis limits for the plot.
    plot_title (str): Additional title text for the plots.
    """
    true_cols = [col for col in results.columns if col.startswith(true_col_prefix)
                 if len(col.split("_"))==3]
    pred_cols = [col for col in results.columns if col.startswith(pred_col_prefix)]

    for true_col in true_cols:
        target = true_col.split('_')[-1]
        related_pred_cols = [col for col in pred_cols if col.endswith(f'_{target}')]

        plt.figure(figsize=(12, 8))
        first_curve = True
        for pred_col in related_pred_cols:
            model_name = pred_col.split('_')[2]  # Assuming format: Y_pred_modelname_xx
            y_true = results[true_col]
            y_pred_proba = results[pred_col]

            thresholds = np.linspace(0, 1, threshold_steps)
            nb_thresholds, net_benefits, all_positive, all_negative =\
                net_benefit_curve(y_true, y_pred_proba, thresholds)

            plt.plot(nb_thresholds, net_benefits, label=f'{model_name}')

            if first_curve:
                # Plot 'all positive' and 'all negative' lines
                plt.plot(nb_thresholds, all_positive, label='All Positive', linestyle='-.', color='black', lw=2)
                plt.plot(nb_thresholds, all_negative, label='All Negative', linestyle='--', color='black', lw=2)
                first_curve=False

        plt.xlabel('Threshold')
        plt.ylabel('Net Benefit')
        plt.title(f'Net Benefit Curves; {plot_title};  {target} ')
        plt.legend()
        plt.grid(True)

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        if output_path:
            plt.savefig(f"{output_path}/net_benefit_curves_{target}.png", dpi=300)
        if ~show_plot:
            plt.close()

def calibration_curve_plot(results: pd.DataFrame = None,
                           true_col_prefix='Y_test',
                           pred_col_prefix='Y_pred',
                           output_path="",
                           n_bins=10,
                           show_plot=False,
                           plot_title=""):
    """
    Plot calibration curves for each target, with separate plots per target.

    Args:
    results (pd.DataFrame): DataFrame containing true values and predictions.
    true_col_prefix (str): Prefix for columns containing true values.
    pred_col_prefix (str): Prefix for columns containing predictions.
    output_path (str): Path to save the output plots.
    n_bins (int): Number of bins for calibration curve.
    """
    true_cols = [col for col in results.columns if col.startswith(true_col_prefix)
                 if len(col.split("_"))==3]
    pred_cols = [col for col in results.columns if col.startswith(pred_col_prefix)]

    for true_col in true_cols:
        target = true_col.split('_')[-1]
        related_pred_cols = [col for col in pred_cols if col.endswith(f'_{target}')]

        plt.figure(figsize=(12, 8))

        for pred_col in related_pred_cols:
            model_name = pred_col.split('_')[2]  # Assuming format: Y_pred_modelname_xx
            y_true = results[true_col]
            y_pred_proba = results[pred_col]

            prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)

            plt.plot(prob_pred, prob_true, marker='o', label=f'{model_name}')

        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title(f'Calibration Curves; {plot_title}; {target}. Uncalibrated')
        plt.legend()
        plt.grid(True)

        if output_path:
            plt.savefig(f"{output_path}/calibration_curves_{target}.png", dpi=300)
        if ~show_plot:
            plt.close()

def calibrator(results: pd.DataFrame,
               index_columns: tuple=('train_index', 'test_index'),
               true_col_prefix='Y_test',
               pred_col_prefix='Y_pred') -> Dict[str, Callable]:
    '''
        Generate calibration functions.

        results: DataFrame with Y_test and Y_pred columns. {true_col_prefix} is post-fixed with _{class_name}
            {pred_col_prefix} is post-fixed with _{model_name}_{class_name}. Fold: the fold number,
            Repeat: the repetition number of the cross-validation. Also contains the index_columns
        index_columns: tuple of column names for train and test indices
        true_col_prefix (str): Prefix for columns containing true values.
        pred_col_prefix (str): Prefix for columns containing predictions.

        -> return a dictionary with calibration functions (in sklearn format), keyed with the model name
    '''
    pass


def net_benefit_curve(y_true, y_pred_proba, thresholds):
    """
    Calculate the net benefit curve for a binary classifier.

    Args:
    y_true (array-like): True binary labels.
    y_pred_proba (array-like): Predicted probabilities for the positive class.
    thresholds (array-like): Threshold values to evaluate.

    Returns:
    tuple: Arrays of thresholds, corresponding net benefit values, all positive net benefit, and all negative net benefit.
    """
    N = len(y_true)
    net_benefits = []
    prevalence = np.mean(y_true)
    all_positive_benefits = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # NB = (TP - FP * T / (1-T)) / N
        net_benefit = (tp - fp * threshold / (1 - threshold)) / N
        net_benefits.append(net_benefit)

        # Calculate net benefit for 'all positive' strategy
        all_positive_benefit = tp/N - threshold / (1 - threshold)*(1-prevalence)
        all_positive_benefits.append(all_positive_benefit)

    # Calculate net benefit for 'all negative' strategy
    all_negative = np.zeros_like(thresholds)

    return thresholds, np.array(net_benefits), np.array(all_positive_benefits), all_negative

def threshold_scores_multiclass(model, X_test, y_test, metric='f1', thresholds=None):
    """
    Optimize the probability thresholds for a multiclass classification model.

    Parameters:
    model : trained model object
        The model should have a predict_proba method.
    X_test : array-like
        The input features of the test set.
    y_test : array-like
        The true labels of the test set.
    metric : str, optional (default='f1')
        The metric to optimize. Options: 'f1', 'precision', 'recall'
    thresholds : array-like, optional (default=None)
        The thresholds to evaluate. If None, np.arange(0.1, 1, 0.01) will be used.

    Returns:
    best_thresholds : list of float
        The thresholds that optimize the specified metric for each class.
    best_scores : list of float
        The best scores achieved for the specified metric for each class.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1, 0.01)

    y_pred_proba = model.predict_proba(X_test)
    n_classes = y_pred_proba.shape[1]

    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    results = defaultdict(list)

    for class_idx in range(n_classes):
        best_threshold = None
        best_score = 0

        for threshold in thresholds:
            y_pred_class = (y_pred_proba[:, class_idx] >= threshold).astype(int)

            if metric == 'f1':
                score = f1_score(y_test_bin[:, class_idx], y_pred_class, average='binary')
            elif metric == 'precision':
                score = precision_score(y_test_bin[:, class_idx], y_pred_class, average='binary')
            elif metric == 'recall':
                score = recall_score(y_test_bin[:, class_idx], y_pred_class, average='binary')
            else:
                raise ValueError("Invalid metric. Choose from 'f1', 'precision', 'recall'")

            results[class_idx].append((threshold, score))
    return results
def apply_thresholds(y_pred_proba, thresholds):
    """
    Apply the optimized thresholds to probability predictions.

    Parameters:
    y_pred_proba : array-like
        The probability predictions from the model.
    thresholds : list of float
        The optimized thresholds for each class.

    Returns:
    y_pred : array-like
        The class predictions after applying the thresholds.
    """
    y_pred = np.zeros(y_pred_proba.shape[0], dtype=int)
    for i in range(y_pred_proba.shape[1]):
        mask = (y_pred_proba[:, i] >= thresholds[i]) & (y_pred == 0)
        y_pred[mask] = i
    return y_pred

def get_best_thresholds(threshold_results):
    """
    Get the best thresholds for each class from the results.

    Parameters:
    threshold_results : dict
        The results from threshold_scores_multiclass.

    Returns:
    best_thresholds : dict
        The best thresholds for each class.
    """
    best_thresholds = {}
    for class_idx, thresholds in threshold_results.items():
        best_threshold, best_score = max(thresholds, key=lambda x: x[1])
        best_thresholds[class_idx] = best_threshold
    return best_thresholds

def training_loop(X:pd.DataFrame=None, Y:pd.DataFrame=None,
                  Splitter: Callable=None,
                  PipeDict: dict=None,
                  use_class_weights: bool=True,
                  num_splits: int=5,
                  num_repeats: int=10,
                  make_df: bool=False,
                  ClassMap: dict=None):
    assert len(ClassMap.keys())>=2, "ClassMap is <2 elements, expected >=2"
    num_classes = len(ClassMap.keys())
    RES_LIST = []
    df_list = []

    indices_df_list = []
    for i, (train_index, test_index) in tqdm.tqdm(enumerate(Splitter.split(X, Y)),
                                                  total=num_splits * num_repeats):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        if use_class_weights == True:
            WeightMap = dict(zip(np.unique(Y_train), np.bincount(Y_train)))
            VectorizedMapping = np.vectorize(WeightMap.get)
            class_counts = 1000 / VectorizedMapping(Y_train)
        else:
            class_counts = None

        # Train models
        kwargs = {
            'gbc': {
                    'GradientBoosting__sample_weight': class_counts
            },
            'rf': {
                    'RandomForest__sample_weight': class_counts
            },
            'xgb': {
                    'XGBoost__sample_weight': class_counts
            },
            'dt': {
                    'DecisionTree__sample_weight': class_counts
            },
            'lr': {
                    'LogisticRegression__sample_weight': class_counts
            }
        }
        for mod in PipeDict.keys():
            PipeDict[mod].fit(X_train, Y_train, **kwargs[mod])

        # Predict on test set
        res_dict = {}
        for mod in PipeDict.keys():
            res_dict[f'Y_pred_{mod}'] = PipeDict[mod].predict_proba(X_test)

        res_dict['Y_test'] = Y_test
        res_dict['Fold'] = i % num_splits
        res_dict['Repeat'] = i // num_splits


        RES_LIST.append(
            res_dict.copy()
        )
        if make_df:
            indices_df = pd.DataFrame()
            indices_df.index = np.arange(0, X.shape[0], 1)
            indices_df['Fold'] = i % num_splits
            indices_df['Repeat'] = i // num_splits
            indices_df['Dataset'] = 'n/a'
            indices_df.loc[train_index, 'Dataset'] = 'Train'
            indices_df.loc[test_index, 'Dataset'] = 'Test'

            res_dict['Fold'] = len(Y_test) * [res_dict['Fold']]
            res_dict['Repeat'] = len(Y_test) * [res_dict['Repeat']]

            for i in range(num_classes):
                res_dict[f'Y_test_{ClassMap[i]}'] = (Y_test == i).astype(int)

                indices_df.loc[train_index, f'True_{ClassMap[i]}'] = (Y_train == i).astype(int)
                indices_df.loc[test_index, f'True_{ClassMap[i]}'] = (Y_test == i).astype(int)

                for mod in PipeDict.keys():
                    indices_df[f'Proba_{mod}_{ClassMap[i]}'] = np.nan

                    res_dict[f'Y_pred_{mod}_{ClassMap[i]}'] = res_dict[f'Y_pred_{mod}'][:, i]
                    Train_probas = PipeDict[mod].predict_proba(X_train)
                    indices_df.loc[train_index, f'Proba_{mod}_{ClassMap[i]}'] =\
                        Train_probas[:, i]
                    indices_df.loc[test_index, f'Proba_{mod}_{ClassMap[i]}'] =\
                        res_dict[f'Y_pred_{mod}'][:, i]

            for mod in PipeDict.keys():
                res_dict.pop(f'Y_pred_{mod}')
            df_list.append(pd.DataFrame(res_dict))

            indices_df = indices_df.assign(INDEX=indices_df.index)
            indices_df_list.append(indices_df)

    if make_df:
        df = pd.concat(df_list, ignore_index=True)
        inds_df  = pd.concat(indices_df_list, ignore_index=True)
    else:
        df = None
        inds_df = None

    return RES_LIST, df, inds_df


def make_plots(results,
               labelcoder,
               n_classes,
               colors,
               ClassMap,
               output_map=None,
               show_plot=False,
               mod_name: Dict[str, str]={'rf': 'Random Forest',
                         'gbc': 'Gradient Boosted Classifier',
                         'xgb': 'Extreme Gradient Boosted Classifier',
                         'dt': 'Decision Tree',
                         'lr': 'Elasticnet Logistic Regression'},
               models: List[str] = ['rf', 'gbc', 'xgb', 'dt', 'lr'],
               plot_title=""):
    assert output_map is not None, "output_map is required"
    os.makedirs(output_map, exist_ok=True)
    plots = []

    for i in range(n_classes):
        fig, ax = plt.subplots()
        mean_fpr = np.linspace(0, 1, 100)

        model_data = {model: {'tprs': [], 'aucs': []} for model in models}

        for res in results:
            y_test_bin = labelcoder.transform(res['Y_test'])
            ip = 1 if y_test_bin.shape[1] == 1 else i

            for model in model_data:
                pred_key = f'Y_pred_{model}'
                if pred_key in res:
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], res[pred_key][:, ip])
                    interp_tpr = np.interp(mean_fpr, fpr, tpr)
                    interp_tpr[0] = 0.0
                    model_data[model]['tprs'].append(interp_tpr)
                    model_data[model]['aucs'].append(auc(fpr, tpr))

        for idx, (model, data) in enumerate(model_data.items()):
            mean_tpr = np.mean(data['tprs'], axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_tpr = np.std(data['tprs'], axis=0)
            ci_tpr = stats.norm.ppf(0.975) * std_tpr / np.sqrt(len(data['tprs']))

            ax.plot(mean_fpr, mean_tpr, color=colors[idx], lw=2,
                    label=f'{mod_name[model]} (mean AUC = {mean_auc:0.2f})')
            ax.fill_between(mean_fpr, mean_tpr - ci_tpr, mean_tpr + ci_tpr, color=colors[idx], alpha=0.2)

        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')

        class_string = f" {ClassMap[0]}/{ClassMap[1]}" if n_classes == 2 else f"Class {ClassMap[i]}"
        ax.set_title(f'ROC Curve; {plot_title}; {class_string}')
        ax.legend(loc="lower right")

        if show_plot:
            plt.show()

        filename = os.path.join(output_map, f'{ClassMap[i]}_ROC.png')
        fig.savefig(filename, dpi=300)
        plt.close(fig)

        plots.append(fig)
        if n_classes == 2:
            return plots
    return plots

from typing import List, Callable, Dict
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

def get_performance(results: List[Dict] = None,
                    threshold: float = 0.5,
                    ClassMap: Dict[int, str] = None,
                    mod_name: Dict[str, str] = {'rf': 'Random Forest',
                                                'gbc': 'Gradient Boosted Classifier',
                                                'xgb': 'Extreme Gradient Boosted Classifier',
                                                'dt': 'Decision Tree',
                                                'lr': 'Elasticnet Logistic Regression'},
                    models: List[str] = ['rf', 'gbc', 'xgb', 'dt', 'lr'],
                    binarizer: Callable = None) -> List[Dict]:
    perf_list = []
    n_classes = len(ClassMap)
    SPEC = lambda tn, fp: tn / (tn + fp) if (tn + fp) > 0 else 0

    for res in results:
        Y_test_bin = binarizer.transform(res['Y_test'])

        for i in range(n_classes):
            class_string = f"{ClassMap[0]}/{ClassMap[1]}" if n_classes == 2 else ClassMap[i]
            ip = 1 if Y_test_bin.shape[1] == 1 else i

            for model_key, model_name in mod_name.items():
                if model_key in models:
                    Y_pred = res[f'Y_pred_{model_key}']
                    Y_pred_bin = np.where(Y_pred[:, ip] > threshold, 1, 0)

                    cm = confusion_matrix(Y_test_bin[:, i], Y_pred_bin)
                    tn, fp, fn, tp = cm.ravel()

                    res_dict = {
                        'f1': f1_score(Y_test_bin[:, i], Y_pred_bin, zero_division=np.nan),
                        'precision': precision_score(Y_test_bin[:, i], Y_pred_bin, zero_division=np.nan),
                        'recall': recall_score(Y_test_bin[:, i], Y_pred_bin, zero_division=np.nan),
                        'specificity': SPEC(tn, fp),
                        'tn': tn,
                        'tp': tp,
                        'fn': fn,
                        'fp': fp,
                        'model': model_key.upper(),
                        'Class': class_string,
                        'Fold': res['Fold'],
                        'Repeat': res['Repeat']
                    }
                    perf_list.append(res_dict)

            if n_classes == 2:
                break

    return perf_list


def create_feature_combinations(df: pd.DataFrame,
                                lambda_functions: Dict[str, Callable[[pd.Series, pd.Series], pd.Series]]) -> Tuple[
    pd.DataFrame, List[str]]:
    # Create a copy of the original dataframe
    result_df = df.copy()
    new_column_names = []
    new_columns_data = {}

    # Iterate through all pairs of columns
    for i, col1 in tqdm.tqdm(enumerate(df.columns)):
        for j, col2 in enumerate(df.columns[i + 1:], start=i + 1):
            # Apply each lambda function to the column pair
            for func_name, func in lambda_functions.items():
                new_col_name = f"{col1}_{col2}_{func_name}"
                new_columns_data[new_col_name] = func(df[col1], df[col2])
                new_column_names.append(new_col_name)

    # Concatenate all new columns at once
    new_columns_df = pd.DataFrame(new_columns_data)
    result_df = pd.concat([result_df, new_columns_df], axis=1)

    return result_df, new_column_names


# Calibration Error
def ECEs(Y_true, Y_pred, nbins=15):
    # see e.g. guo2017calibration
    # slightly adapted
    counts, leftboundary = np.histogram(Y_pred, density=False, bins=nbins)
    rightboundary = leftboundary[1:]

    totnum = Y_pred.shape[0]
    pmc = []
    rmc = []
    emc = []
    for k, _count in enumerate(counts):
        gte_left = Y_pred >= leftboundary[k]
        if k == len(counts) - 1:
            st_right = Y_pred <= rightboundary[k]
        else:
            st_right = Y_pred < rightboundary[k]
        indcs = np.argwhere(gte_left & st_right)[:, 0]

        Ypc = Y_pred[indcs]
        Ytc = Y_true[indcs]

        Ci = np.mean(Ypc)
        Ai = np.mean(Ytc)  # np.mean((Ypc>0.5).astype(int)==Ytc)

        pmc.append(np.abs(Ai - Ci))
        rmc.append(_count / totnum * (Ai - Ci) ** 2)
        emc.append(_count / totnum * np.sqrt(np.abs(Ai - Ci)))
    ece = sum(emc)
    mce = max(pmc)
    rmsce = np.sqrt(sum(rmc))
    return 1 - ece, 1 - mce, 1 - rmsce


def R2C(Y_true, Y_pred, nbins=15, weighted=True):
    counts, leftboundary = np.histogram(Y_pred, density=False, bins=nbins)
    rightboundary = leftboundary[1:]

    totnum = len(Y_pred)
    emc = []
    tot = []
    Ytm = np.mean(Y_true)
    for k, _count in enumerate(counts):
        gte_left = Y_pred >= leftboundary[k]
        if k == len(counts) - 1:
            st_right = Y_pred <= rightboundary[k]
        else:
            st_right = Y_pred < rightboundary[k]
        indcs = np.argwhere(gte_left & st_right)[:, 0]

        Ypc = Y_pred[indcs]
        Ytc = Y_true[indcs]

        Ci = np.mean(Ypc)
        Ai = np.mean(Ytc)
        if weighted:
            emc.append(_count / totnum * (Ai - Ci) ** 2)
            tot.append(_count / totnum * (Ai - Ytm) ** 2)
        else:
            emc.append((Ai - Ci) ** 2)
            tot.append((Ai - Ytm) ** 2)
    ece = np.nansum(emc)
    tots = np.nansum(tot)
    return 1 - ece / tots


def create_calibration_plots(df: pd.DataFrame = None,
                             ebins: int = 8,
                             cbins: int = 15,
                             write_out: bool = False,
                             output_path: str = None,
                             mod_name: dict={
                                        'LR': 'Logistic Regression',
                                        'XGB': 'eXtreme Gradient Boosting',
                                        'customDT': 'Custom Decision Tree',
                                        'normalDT': 'Normal Decision Tree',
                                    }):
    pred_strings = [c for c in df.columns if c.startswith('Y_pred')]
    Classes = set([s.split("_")[3] for s in pred_strings])
    Models = set([s.split("_")[2] for s in pred_strings])

    for cl in tqdm.tqdm(Classes):
        for mod in Models:
            Yt = df.loc[df.Dataset == 'train', f'Y_true_{cl}']
            Yp = df.loc[df.Dataset == 'train', f'Y_pred_{mod}_{cl}']
            cCurveTrain = calibration_curve(Yt, Yp, strategy='quantile', n_bins=cbins)

            ECEtrain, _, _ = ECEs(Yt.values, Yp.values, nbins=ebins)
            R2Ctrain = R2C(Yt.values, Yp.values, nbins=ebins)

            Yt = df.loc[df.Dataset == 'test', f'Y_true_{cl}']
            Yp = df.loc[df.Dataset == 'test', f'Y_pred_{mod}_{cl}']
            cCurveTest = calibration_curve(Yt, Yp, strategy='quantile', n_bins=cbins)

            ECEtest, _, _ = ECEs(Yt.values, Yp.values, nbins=ebins)
            R2Ctest = R2C(Yt.values, Yp.values, nbins=ebins)

            fig, ax = plt.subplots(ncols=2, figsize=(19, 7))
            ax[0].scatter(cCurveTrain[0], cCurveTrain[1], s=100, label='Train')
            ax[0].scatter(cCurveTest[0], cCurveTest[1], s=100, label='Test')
            ax[0].plot([0, 1], [0, 1], color='black')
            ax[0].legend()
            ax[0].set_xlabel('Model probability', size=20)
            ax[0].set_ylabel('Actual probability', size=20)
            ax[0].legend(prop={'size': 20})

            df.loc[df.Dataset == 'train', f'Y_pred_{mod}_{cl}'].hist(bins=cbins, histtype='step',
                                                                     lw=3, density=True, label='Train',
                                                                     ax=ax[1])
            df.loc[df.Dataset == 'test', f'Y_pred_{mod}_{cl}'].hist(bins=cbins, histtype='step',
                                                                    lw=3, density=True, label='Test',
                                                                    ax=ax[1])
            ax[1].legend(prop={'size': 20})
            ax[1].set_xlabel('Model probability', size=20)
            ax[1].set_ylabel('Density', size=20)

            fig.suptitle(
                f'{mod_name[mod]} calibration: before re-calibration. ECE train/test: {round(ECEtrain, 2)}, {round(ECEtest, 2)}, R2 train/test: {round(R2Ctrain, 2)}, {round(R2Ctest, 2)}')
            plt.tight_layout()
            if write_out:
                plt.savefig(os.path.join(output_path, f'CustomTree_CalibrationPlot_{cl}_{mod}.svg'), dpi=300)
                plt.close(fig)