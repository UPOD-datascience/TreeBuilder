import numpy as np
from scipy import stats
import pandas as pd
import pickle
import os
import tqdm

from typing import List, Callable, Dict, Tuple, Literal

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn.calibration import calibration_curve
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.calibration import IsotonicRegression, _SigmoidCalibration
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, recall_score, precision_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import precision_score, f1_score

from numpy import interp


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
                           ylim=[-0.1, 0.5],
                           show_plot=False,
                           file_suffix="",
                           calibrated=False,
                           dataset="test",
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
    if calibrated:
        pred_cols = [col for col in results.columns if col.startswith(pred_col_prefix) & col.endswith("_calibrated_mean")]
    else:
        pred_cols = [col for col in results.columns if col.startswith(pred_col_prefix)]

    for true_col in true_cols:
        target = true_col.split('_')[-1]
        if calibrated:
            related_pred_cols = [col for col in pred_cols if (f'_{target}' in col) & ('_calibrated' in col)]
        else:
            related_pred_cols = [col for col in pred_cols if col.endswith(f'_{target}')]

        plt.figure(figsize=(12, 8))
        first_curve = True
        for pred_col in related_pred_cols:
            model_name = pred_col.split('_')[2]  # Assuming format: Y_pred_modelname_xx
            y_true = results.loc[results.Dataset == dataset, true_col]
            y_pred_proba = results.loc[results.Dataset == dataset, pred_col]

            #print(model_name, y_true.shape, y_pred_proba.shape)

            thresholds = np.linspace(0, 1, threshold_steps)
            nb_thresholds, net_benefits, all_positive, all_negative =\
                net_benefit_curve(y_true, y_pred_proba, thresholds[:-1])

            plt.plot(nb_thresholds, net_benefits, label=f'{model_name}')

            if first_curve:
                # Plot 'all positive' and 'all negative' lines
                plt.plot(nb_thresholds, all_positive, label='All Positive', linestyle='-.', color='black', lw=2)
                plt.plot(nb_thresholds, all_negative, label='All Negative', linestyle='--', color='black', lw=2)
                first_curve=False

        plt.xlabel('Threshold')
        plt.ylabel('Net Benefit')
        plt.title(f'Net Benefit Curves. {plot_title}. Target={target} ')
        plt.legend()
        plt.grid(True)

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            max_y = np.max(net_benefits)
            if max_y > ylim[1]:
                ylim[1] = max_y
            plt.ylim(ylim)

        if output_path:
            plt.savefig(f"{output_path}/net_benefit_curves_{target}{file_suffix}.svg", dpi=300)
        if ~show_plot:
            plt.close()
        else:
            plt.show()

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
        plt.title(f'Calibration Curves; {plot_title}; {target}.')
        plt.legend()
        plt.grid(True)

        if output_path:
            plt.savefig(f"{output_path}/calibration_curves_{target}.png", dpi=300)
        if ~show_plot:
            plt.close()


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
            ci_tpr = stats.norm.ppf(0.9) * std_tpr / np.sqrt(len(data['tprs']))

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
                                    },
                             suffix=''):
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
            Yp = df.loc[df.Dataset == 'test', f'Y_pred_{mod}_{cl}{suffix}']
            cCurveTest = calibration_curve(Yt, Yp, strategy='quantile', n_bins=cbins)

            ECEtest, _, _ = ECEs(Yt.values, Yp.values, nbins=ebins)
            R2Ctest = R2C(Yt.values, Yp.values, nbins=ebins)

            fig, ax = plt.subplots(ncols=2, figsize=(19, 7))
            ax[0].plot(cCurveTrain[0], cCurveTrain[1], marker='o', label='Train', lw=2)
            ax[0].plot(cCurveTest[0], cCurveTest[1], marker='o', label='Test', lw=2)
            ax[0].plot([0, 1], [0, 1], color='red', linestyle='--', lw=2)
            ax[0].legend()
            ax[0].set_xlabel('Model probability', size=20)
            ax[0].set_ylabel('Actual probability', size=20)
            ax[0].legend(prop={'size': 20})

            df.loc[df.Dataset == 'train', f'Y_pred_{mod}_{cl}'].hist(bins=cbins, histtype='step',
                                                                     lw=3, density=True, label='Train',
                                                                     ax=ax[1])
            df.loc[df.Dataset == 'test', f'Y_pred_{mod}_{cl}{suffix}'].hist(bins=cbins, histtype='step',
                                                                    lw=3, density=True, label='Test',
                                                                    ax=ax[1])
            ax[1].legend(prop={'size': 20})
            ax[1].set_xlabel('Model probability', size=20)
            ax[1].set_ylabel('Density', size=20)

            fig.suptitle(
                f'{mod_name[mod]} calibration: ECE train/test: {round(ECEtrain, 2)}, {round(ECEtest, 2)}, R2 train/test: {round(R2Ctrain, 2)}, {round(R2Ctest, 2)}')
            plt.tight_layout()
            if write_out:
                plt.savefig(os.path.join(output_path, f'CustomTree_CalibrationPlot_{cl}_{mod}{suffix}.svg'), dpi=300)
                plt.close(fig)


def calibrater(y_true, y_preds, how: Literal = ['isotonic', 'linear', 'sigmoid']):
    if how == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
    elif how == 'linear':
        calibrator = LinearRegression(positive=True)
    elif how == 'sigmoid':
        calibrator = _SigmoidCalibration()
    else:
        raise ValueError("method should be one of isotonic, linear or sigmoid")

    calibrator.fit(y_preds, y_true)
    return calibrator

def add_calibrated_values(df, how='isotonic'):
    CALIBRATOR = defaultdict(lambda: defaultdict(list))

    pred_strings = [c for c in df.columns if c.startswith('Y_pred')]
    Classes = set([s.split("_")[3] for s in pred_strings])
    Models = set([s.split("_")[2] for s in pred_strings])
    Folds = df.Fold.unique().tolist()
    Repeats = df.Repeat.unique().tolist()

    for _class in Classes:
        for _mod in Models:
            new_col = f'Y_pred_{_mod}_{_class}_calibrated_mean'
            new_col_std = f'Y_pred_{_mod}_{_class}_calibrated_std'
            df[new_col] = np.nan
            df[new_col_std] = np.nan
            for _Repeat in Repeats:
                tmp_calibrator_list = []
                for _Fold in Folds:
                    conds = (df.Repeat == _Repeat) & (df.Fold == _Fold) & (df.Dataset == 'test')
                    Y_true = df.loc[conds, f'Y_true_{_class}'].values
                    Y_pred = df.loc[conds, f'Y_pred_{_mod}_{_class}'].values

                    Calibration_model = calibrater(Y_true, Y_pred, how=how)

                    tmp_calibrator_list.append(Calibration_model)
                    CALIBRATOR[_class][_mod].append(Calibration_model)
                # now we collect the calibrated probas for all folds based on all the calibrations
                _conds = (df.Repeat == _Repeat) & (df.Dataset == 'test')
                calibrated_list = []
                _y_preds = df.loc[_conds, f'Y_pred_{_mod}_{_class}'].values
                for _Fold, _calibrater in enumerate(tmp_calibrator_list):
                    calibrated_list.append(_calibrater.predict(_y_preds))
                df.loc[_conds, new_col] = np.mean(calibrated_list, axis=0)
                df.loc[_conds, new_col_std] = np.std(calibrated_list, axis=0)
    return df, CALIBRATOR


def make_roc_plots(df,
                   OutPath: str=None,
                   FoldColumn: str='Fold',
                   RepeatColumn: str='Repeat',
                   DataSetColumn: str='Dataset',
                   n_thresholds: int=50,
                   Target: Literal=['Heart Axis', 'Muscle', 'Conduction'],
                   plot_title: str="",
                   suffix: str='',
                   mod_name: dict = {
                       'LR': 'Logistic Regression',
                       'XGB': 'eXtreme Gradient Boosting',
                       'customDT': 'Custom Decision Tree',
                       'normalDT': 'Normal Decision Tree',
                   }
                   ):

    pred_strings = [c for c in df.columns if c.startswith('Y_pred')]
    Classes = set([s.split("_")[3] for s in pred_strings])
    Models = set([s.split("_")[2] for s in pred_strings])

    for Class in Classes:
        plt.figure(figsize=(10, 8))
        for Model in Models:
            _OutPath = os.path.join(OutPath, f"ROC_curve_{Class}_{Model}{suffix}.svg")
            _Title = f"Target: {Target}, Class: {Class}, Model: {Model}"
            mean_auc, std_auc, roc_curve_data = _make_roc_plot(df,
                                                       TestColumn=f'Y_true_{Class}',
                                                       PredColumn=f'Y_pred_{Model}_{Class}{suffix}',
                                                       FoldColumn=FoldColumn,
                                                       RepeatColumn=RepeatColumn,
                                                       DataSetColumn=DataSetColumn,
                                                       OutPath=_OutPath,
                                                       n_thresholds=n_thresholds,
                                                       plot_title=_Title,
                                                       return_curve=True)

            plt.plot(roc_curve_data[0], roc_curve_data[1],
                         label=f"{mod_name[Model]}. AUC={round(mean_auc,2)} ± {round(std_auc,2)}", lw=2)
            line_color = plt.gca().lines[-1].get_color()
            tprs_lower = np.maximum(roc_curve_data[1] - roc_curve_data[2], 0)
            tprs_upper = np.minimum(roc_curve_data[1] + roc_curve_data[2], 1)
            plt.fill_between(roc_curve_data[0], tprs_lower, tprs_upper, color=line_color, alpha=.1)  # label=f'± 1 std. dev.'

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', label='Chance', alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', size=20)
        plt.ylabel('True Positive Rate', size=20)
        plt.title(f'ROC curve. Target: {Target}, Class: {Class}. {plot_title}', size=20)
        plt.legend(loc="lower right", prop={'size': 20})
        plt.savefig(os.path.join(OutPath,f"ROC_{Target}_{Class}{suffix}.svg"), dpi=300)
        plt.close()
    return True

def _make_roc_plot(df: pd.DataFrame, TestColumn: str = 'Y_true',
                   PredColumn: str = 'Y_pred',
                   RepeatColumn: str = 'Repeat',
                   FoldColumn: str = 'Fold',
                   DataSetColumn: str = 'Dataset',
                   OutPath: str = None,
                   n_thresholds: int = 50,
                   return_curve: bool = True,
                   plot_title=""):
    '''
    Make ROC plots from a dataframe that contains multiple folds/repeat of test predictions

    df: pandas Datatrame
    TestColumn: str, column name for true test labels
    PredColumn: str, column name for proba of predicted test labels
    RepeatColumn: str, column name for repeat number of cross-validation
    FoldColumn: str, column name for fold number of cross-validation
    DataSetColumn: str, column name for dataset (train/test)
    OutPath: str, output directory for ROC plots (if empty, just display the column)
    n_thresholds: int, number of thresholds to use for ROC curve (default: 100)
    '''

    # Filter for test set data
    test_df = df[df[DataSetColumn] == 'test']

    # Get unique repeats
    repeats = test_df[RepeatColumn].unique()

    plt.figure(figsize=(10, 8))

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, n_thresholds)

    for r in repeats:
        repeat_df = test_df[test_df[RepeatColumn] == r]

        for f in repeat_df[FoldColumn].unique():
            fold_df = repeat_df[repeat_df[FoldColumn] == f]

            fpr, tpr, _ = roc_curve(fold_df[TestColumn], fold_df[PredColumn])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=1, alpha=0.1, color='black') #label=f'ROC fold {f}, repeat {r} (AUC = {roc_auc:.2f})')


            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(roc_auc)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})',
             lw=3, alpha=1)

    std_tpr = np.std(tprs, axis=0)
    ci_tpr = stats.norm.ppf(0.99) * std_tpr / np.sqrt(len(tprs))

    tprs_upper = np.minimum(mean_tpr + ci_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - ci_tpr, 0)

    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2) #label=f'± 1 std. dev.'

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', size=20)
    plt.ylabel('True Positive Rate', size=20)
    plt.title(f'ROC curve. {plot_title}', size=20)
    plt.legend(loc="lower right", prop={'size': 20})

    if OutPath:
        plt.savefig(OutPath, dpi=300)
    else:
        plt.show()

    plt.close()

    if return_curve:
        return mean_auc, std_auc, (mean_fpr, mean_tpr, ci_tpr)
    else:
        return mean_auc, std_auc, None

def make_precisionRecall_plots(df,
                               OutPath: str=None,
                               FoldColumn: str='Fold',
                               RepeatColumn: str='Repeat',
                               DataSetColumn: str='Dataset',
                               n_thresholds: int=50,
                               Target: Literal=['Heart Axis', 'Muscle', 'Conduction'],
                               plot_title: str="",
                               suffix: str='',
                               mod_name: dict = {
                                   'LR': 'Logistic Regression',
                                   'XGB': 'eXtreme Gradient Boosting',
                                   'customDT': 'Custom Decision Tree',
                                   'normalDT': 'Normal Decision Tree',
                               }
                               ):
    pred_strings = [c for c in df.columns if c.startswith('Y_pred')]
    Classes = set([s.split("_")[3] for s in pred_strings])
    Models = set([s.split("_")[2] for s in pred_strings])

    for Class in Classes:
        plt.figure(figsize=(10, 8))
        for Model in Models:
            _OutPath = os.path.join(OutPath, f"PrecisionRecall_curve_{Class}_{Model}{suffix}.svg")
            _Title = f"Target: {Target}, Class: {Class}, Model: {Model}"
            mean_auprc, std_auprc, pr_curve_data = _make_precisionRecall_plot(df,
                                                       TestColumn=f'Y_true_{Class}',
                                                       PredColumn=f'Y_pred_{Model}_{Class}{suffix}',
                                                       FoldColumn=FoldColumn,
                                                       RepeatColumn=RepeatColumn,
                                                       DataSetColumn=DataSetColumn,
                                                       OutPath=_OutPath,
                                                       n_thresholds=n_thresholds,
                                                       plot_title=_Title,
                                                       return_curve=True)

            plt.plot(pr_curve_data[0], pr_curve_data[1],
                     label=f"{mod_name[Model]}. AUPRC={round(mean_auprc,2)} ± {round(std_auprc,2)}", lw=2)
            line_color = plt.gca().lines[-1].get_color()
            precision_lower = np.maximum(pr_curve_data[1] - pr_curve_data[2], 0)
            precision_upper = np.minimum(pr_curve_data[1] + pr_curve_data[2], 1)
            plt.fill_between(pr_curve_data[0], precision_lower, precision_upper, color=line_color, alpha=.1)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall', size=20)
        plt.ylabel('Precision', size=20)
        plt.title(f'Precision-Recall curve. Target: {Target}, Class: {Class}. {plot_title}', size=20)
        plt.legend(loc="lower left", prop={'size': 20})
        plt.savefig(os.path.join(OutPath,f"PrecisionRecall_{Target}_{Class}{suffix}.svg"), dpi=300)
        plt.close()
    return True


def _make_precisionRecall_plot(df: pd.DataFrame, TestColumn: str = 'Y_true',
                               PredColumn: str = 'Y_pred',
                               RepeatColumn: str = 'Repeat',
                               FoldColumn: str = 'Fold',
                               DataSetColumn: str = 'Dataset',
                               OutPath: str = None,
                               n_thresholds: int = 50,
                               return_curve: bool = True,
                               plot_title=""):
    # Filter for test set data
    test_df = df[df[DataSetColumn] == 'test']

    # Get unique repeats
    repeats = test_df[RepeatColumn].unique()

    plt.figure(figsize=(10, 8))

    precisions = []
    auprcs = []
    mean_recall = np.linspace(0, 1, n_thresholds)

    for r in repeats:
        repeat_df = test_df[test_df[RepeatColumn] == r]

        for f in repeat_df[FoldColumn].unique():
            fold_df = repeat_df[repeat_df[FoldColumn] == f]

            precision, recall, _ = precision_recall_curve(fold_df[TestColumn], fold_df[PredColumn])
            auprc = average_precision_score(fold_df[TestColumn], fold_df[PredColumn])

            plt.plot(recall, precision, lw=1, alpha=0.1, color='black')

            precisions.append(interp(mean_recall, recall[::-1], precision[::-1]))
            auprcs.append(auprc)

    mean_precision = np.mean(precisions, axis=0)
    mean_auprc = np.mean(auprcs)
    std_auprc = np.std(auprcs)
    plt.plot(mean_recall, mean_precision, color='b',
             label=f'Mean PR (AUPRC = {mean_auprc:.2f} ± {std_auprc:.2f})',
             lw=3, alpha=1)

    std_precision = np.std(precisions, axis=0)
    ci_precision = stats.norm.ppf(0.99) * std_precision / np.sqrt(len(precisions))

    precision_upper = np.minimum(mean_precision + ci_precision, 1)
    precision_lower = np.maximum(mean_precision - ci_precision, 0)

    plt.fill_between(mean_recall, precision_lower, precision_upper, color='grey', alpha=.2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall', size=20)
    plt.ylabel('Precision', size=20)
    plt.title(f'Precision-Recall curve. {plot_title}', size=20)
    plt.legend(loc="lower left", prop={'size': 20})

    if OutPath:
        plt.savefig(OutPath, dpi=300)
    else:
        plt.show()

    plt.close()

    if return_curve:
        return mean_auprc, std_auprc, (mean_recall, mean_precision, ci_precision)
    else:
        return mean_auprc, std_auprc, None


def make_recall_plots(df,
                      OutPath: str = None,
                      FoldColumn: str = 'Fold',
                      RepeatColumn: str = 'Repeat',
                      DataSetColumn: str = 'Dataset',
                      n_thresholds: int = 50,
                      Target: Literal=['Heart Axis', 'Muscle', 'Conduction'],
                      plot_title: str = "",
                      suffix: str = '',
                      mod_name: dict = {
                          'LR': 'Logistic Regression',
                          'XGB': 'eXtreme Gradient Boosting',
                          'customDT': 'Custom Decision Tree',
                          'normalDT': 'Normal Decision Tree',
                      },
                      only_data: bool=False
                      ):
    '''
    Function to create plots with proba_thresholds versus recall-score.
    '''
    pred_strings = [c for c in df.columns if c.startswith('Y_pred')]
    Classes = set([s.split("_")[3] for s in pred_strings])
    Models = set([s.split("_")[2] for s in pred_strings])

    for Class in Classes:
        data_list = []
        plt.figure(figsize=(10, 8))
        for Model in Models:
            _OutPath = os.path.join(OutPath, f"Recall_curve_{Class}_{Model}{suffix}.svg")
            _Title = f"Target: {Target}, Class: {Class}, Model: {Model}"
            mean_recall, std_recall, recall_curve_data = _make_recall_plot(df,
                                                                           TestColumn=f'Y_true_{Class}',
                                                                           PredColumn=f'Y_pred_{Model}_{Class}{suffix}',
                                                                           FoldColumn=FoldColumn,
                                                                           RepeatColumn=RepeatColumn,
                                                                           DataSetColumn=DataSetColumn,
                                                                           OutPath=_OutPath,
                                                                           n_thresholds=n_thresholds,
                                                                           plot_title=_Title,
                                                                           return_curve=True)
            data_list.append({'thresholds': list(recall_curve_data[0]),
                              'mean_value': list(recall_curve_data[1]),
                              'std_value': list(recall_curve_data[2]),
                              'Model': len(recall_curve_data[0])*[Model]
                            })

            if only_data == False:
                plt.plot(recall_curve_data[0], recall_curve_data[1],
                         label=f"{mod_name[Model]}. Mean Recall={round(mean_recall, 2)} ± {round(std_recall, 2)}", lw=2)
                line_color = plt.gca().lines[-1].get_color()
                recall_lower = np.maximum(recall_curve_data[1] - recall_curve_data[2], 0)
                recall_upper = np.minimum(recall_curve_data[1] + recall_curve_data[2], 1)
                plt.fill_between(recall_curve_data[0], recall_lower, recall_upper, color=line_color, alpha=.1)

        if only_data == False:
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('Probability Threshold', size=20)
            plt.ylabel('Recall', size=20)
            plt.title(f'Recall vs Threshold. Target: {Target}, Class: {Class}. {plot_title}', size=20)
            plt.legend(loc="lower left", prop={'size': 20})
            plt.savefig(os.path.join(OutPath, f"Recall_{Target}_{Class}{suffix}.svg"), dpi=300)
        plt.close()

        data_df = pd.DataFrame()
        for _df in data_list:
            data_df = pd.concat([data_df, pd.DataFrame.from_dict(_df, orient='columns')])
        data_df.to_csv(os.path.join(OutPath, f"Recall_{Target}_{Class}{suffix}_DATA.csv"), sep="\t")
    return True


def _make_recall_plot(df: pd.DataFrame, TestColumn: str = 'Y_true',
                      PredColumn: str = 'Y_pred',
                      RepeatColumn: str = 'Repeat',
                      FoldColumn: str = 'Fold',
                      DataSetColumn: str = 'Dataset',
                      OutPath: str = None,
                      n_thresholds: int = 50,
                      return_curve: bool = True,
                      plot_title="",
                      only_data: bool=False):
    if only_data:
        return_curve=True
        
    test_df = df[df[DataSetColumn] == 'test']
    repeats = test_df[RepeatColumn].unique()

    plt.figure(figsize=(10, 8))

    recalls = []
    thresholds = np.linspace(0, 1, n_thresholds)
    for r in repeats:
        repeat_df = test_df[test_df[RepeatColumn] == r]
        for f in repeat_df[FoldColumn].unique():
            fold_df = repeat_df[repeat_df[FoldColumn] == f]

            fold_recalls = []
            for threshold in thresholds:
                y_pred = (fold_df[PredColumn] >= threshold).astype(int)
                recall = recall_score(fold_df[TestColumn], y_pred)
                fold_recalls.append(recall)
            if only_data == False:
                plt.plot(thresholds, fold_recalls, lw=1, alpha=0.1, color='black')
            recalls.append(fold_recalls)

    mean_recall = np.mean(recalls, axis=0)
    std_recall = np.std(recalls, axis=0)

    if only_data == False:
        plt.plot(thresholds, mean_recall, color='b',
                 label=f'Mean Recall',
                 lw=3, alpha=1)

    recall_upper = np.minimum(mean_recall + std_recall, 1)
    recall_lower = np.maximum(mean_recall - std_recall, 0)

    if only_data == False:
        plt.fill_between(thresholds, recall_lower, recall_upper, color='grey', alpha=.2)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Probability Threshold', size=20)
        plt.ylabel('Recall', size=20)
        plt.title(f'Recall vs Threshold. {plot_title}', size=20)
        plt.legend(loc="lower left", prop={'size': 20})

        if OutPath:
            plt.savefig(OutPath, dpi=300)
        else:
            plt.show()
    plt.close()

    if return_curve:
        return np.mean(mean_recall), np.mean(std_recall), (thresholds, mean_recall, std_recall)
    else:
        return np.mean(mean_recall), np.mean(std_recall), None


def make_npv_plots(df,
                   OutPath: str = None,
                   FoldColumn: str = 'Fold',
                   RepeatColumn: str = 'Repeat',
                   DataSetColumn: str = 'Dataset',
                   n_thresholds: int = 50,
                   Target: Literal=['Heart Axis', 'Muscle', 'Conduction'],
                   plot_title: str = "",
                   suffix: str = '',
                   mod_name: dict = {
                       'LR': 'Logistic Regression',
                       'XGB': 'eXtreme Gradient Boosting',
                       'customDT': 'Custom Decision Tree',
                       'normalDT': 'Normal Decision Tree',
                   },
                   only_data: bool = False
                   ):
    '''
    Function to create plots with proba_thresholds versus negative predictive value.
    '''
    pred_strings = [c for c in df.columns if c.startswith('Y_pred')]
    Classes = set([s.split("_")[3] for s in pred_strings])
    Models = set([s.split("_")[2] for s in pred_strings])

    for Class in Classes:
        data_list = []
        plt.figure(figsize=(10, 8))
        for Model in Models:
            _OutPath = os.path.join(OutPath, f"NPV_curve_{Class}_{Model}{suffix}.svg")
            _Title = f"Target: {Target}, Class: {Class}, Model: {Model}"
            mean_npv, std_npv, npv_curve_data = _make_npv_plot(df,
                                                               TestColumn=f'Y_true_{Class}',
                                                               PredColumn=f'Y_pred_{Model}_{Class}{suffix}',
                                                               FoldColumn=FoldColumn,
                                                               RepeatColumn=RepeatColumn,
                                                               DataSetColumn=DataSetColumn,
                                                               OutPath=_OutPath,
                                                               n_thresholds=n_thresholds,
                                                               plot_title=_Title,
                                                               return_curve=True,
                                                               only_data=only_data)
            data_list.append({'thresholds': list(npv_curve_data[0]),
                              'mean_value': list(npv_curve_data[1]),
                              'std_value': list(npv_curve_data[2]),
                              'Model': len(npv_curve_data[0])*[Model]
                            })

            if only_data == False:
                plt.plot(npv_curve_data[0], npv_curve_data[1],
                         label=f"{mod_name[Model]}. Mean NPV={round(mean_npv, 2)} ± {round(std_npv, 2)}", lw=2)
                line_color = plt.gca().lines[-1].get_color()
                npv_lower = np.maximum(npv_curve_data[1] - npv_curve_data[2], 0)
                npv_upper = np.minimum(npv_curve_data[1] + npv_curve_data[2], 1)
                plt.fill_between(npv_curve_data[0], npv_lower, npv_upper, color=line_color, alpha=.1)

        if only_data == False:
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('Probability Threshold', size=20)
            plt.ylabel('Negative Predictive Value', size=20)
            plt.title(f'NPV vs Threshold. Target: {Target}, Class: {Class}. {plot_title}', size=20)
            plt.legend(loc="lower left", prop={'size': 20})
            plt.savefig(os.path.join(OutPath, f"NPV_{Target}_{Class}{suffix}.svg"), dpi=300)
        plt.close()

        data_df = pd.DataFrame()
        for _df in data_list:
            data_df = pd.concat([data_df, pd.DataFrame.from_dict(_df, orient='columns')])
        data_df.to_csv(os.path.join(OutPath, f"NPV_{Target}_{Class}{suffix}_DATA.csv"), sep="\t")
    return True


def _make_npv_plot(df: pd.DataFrame, TestColumn: str = 'Y_true',
                   PredColumn: str = 'Y_pred',
                   RepeatColumn: str = 'Repeat',
                   FoldColumn: str = 'Fold',
                   DataSetColumn: str = 'Dataset',
                   OutPath: str = None,
                   n_thresholds: int = 50,
                   return_curve: bool = True,
                   plot_title="",
                   only_data: bool = False):
    if only_data:
        return_curve=True

    test_df = df[df[DataSetColumn] == 'test']
    repeats = test_df[RepeatColumn].unique()

    plt.figure(figsize=(10, 8))

    npvs = []
    thresholds = np.linspace(0, 1, n_thresholds)

    for r in repeats:
        repeat_df = test_df[test_df[RepeatColumn] == r]

        for f in repeat_df[FoldColumn].unique():
            fold_df = repeat_df[repeat_df[FoldColumn] == f]

            fold_npvs = []
            for threshold in thresholds:
                y_pred = (fold_df[PredColumn] >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(fold_df[TestColumn], y_pred).ravel()
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                fold_npvs.append(npv)
            if only_data == False:
                plt.plot(thresholds, fold_npvs, lw=1, alpha=0.1, color='black')
            npvs.append(fold_npvs)

    mean_npv = np.mean(npvs, axis=0)
    std_npv = np.std(npvs, axis=0)

    if only_data == False:
        plt.plot(thresholds, mean_npv, color='b',
                 label=f'Mean NPV',
                 lw=3, alpha=1)

    npv_upper = np.minimum(mean_npv + std_npv, 1)
    npv_lower = np.maximum(mean_npv - std_npv, 0)

    if only_data == False:
        plt.fill_between(thresholds, npv_lower, npv_upper, color='grey', alpha=.2)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Probability Threshold', size=20)
        plt.ylabel('Negative Predictive Value', size=20)
        plt.title(f'NPV vs Threshold. {plot_title}', size=20)
        plt.legend(loc="lower left", prop={'size': 20})

        if OutPath:
            plt.savefig(OutPath, dpi=300)
        else:
            plt.show()
    plt.close()

    if return_curve:
        return np.mean(mean_npv), np.mean(std_npv), (thresholds, mean_npv, std_npv)
    else:
        return np.mean(mean_npv), np.mean(std_npv), None


def make_precision_plots(df,
                         OutPath: str = None,
                         FoldColumn: str = 'Fold',
                         RepeatColumn: str = 'Repeat',
                         DataSetColumn: str = 'Dataset',
                         n_thresholds: int = 50,
                         Target: Literal=['Heart Axis', 'Muscle', 'Conduction'],
                         plot_title: str = "",
                         suffix: str = '',
                         mod_name: dict = {
                             'LR': 'Logistic Regression',
                             'XGB': 'eXtreme Gradient Boosting',
                             'customDT': 'Custom Decision Tree',
                             'normalDT': 'Normal Decision Tree',
                         },
                         only_data:bool = False
                         ):
    '''
    Function to create plots with proba_thresholds versus precision (PPV).
    '''
    pred_strings = [c for c in df.columns if c.startswith('Y_pred')]
    Classes = set([s.split("_")[3] for s in pred_strings])
    Models = set([s.split("_")[2] for s in pred_strings])

    for Class in Classes:
        data_list = []
        plt.figure(figsize=(10, 8))
        for Model in Models:
            _OutPath = os.path.join(OutPath, f"Precision_curve_{Class}_{Model}{suffix}.svg")
            _Title = f"Target: {Target}, Class: {Class}, Model: {Model}"
            mean_precision, std_precision, precision_curve_data = _make_precision_plot(df,
                                                                                       TestColumn=f'Y_true_{Class}',
                                                                                       PredColumn=f'Y_pred_{Model}_{Class}{suffix}',
                                                                                       FoldColumn=FoldColumn,
                                                                                       RepeatColumn=RepeatColumn,
                                                                                       DataSetColumn=DataSetColumn,
                                                                                       OutPath=_OutPath,
                                                                                       n_thresholds=n_thresholds,
                                                                                       plot_title=_Title,
                                                                                       return_curve=True,
                                                                                       only_data=only_data)
            data_list.append({'thresholds': list(precision_curve_data[0]),
                              'mean_value': list(precision_curve_data[1]),
                              'std_value': list(precision_curve_data[2]),
                              'Model': len(precision_curve_data[0])*[Model]
                              })

            if only_data == False:
                plt.plot(precision_curve_data[0], precision_curve_data[1],
                         label=f"{mod_name[Model]}. Mean Precision={round(mean_precision, 2)} ± {round(std_precision, 2)}",
                         lw=2)
                line_color = plt.gca().lines[-1].get_color()
                precision_lower = np.maximum(precision_curve_data[1] - precision_curve_data[2], 0)
                precision_upper = np.minimum(precision_curve_data[1] + precision_curve_data[2], 1)
                plt.fill_between(precision_curve_data[0], precision_lower, precision_upper, color=line_color, alpha=.1)

        if only_data == False:
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('Probability Threshold', size=20)
            plt.ylabel('Precision', size=20)
            plt.title(f'Precision vs Threshold. Target: {Target}, Class: {Class}. {plot_title}', size=20)
            plt.legend(loc="lower left", prop={'size': 20})
            plt.savefig(os.path.join(OutPath, f"Precision_{Target}_{Class}{suffix}.svg"), dpi=300)
        plt.close()

        data_df = pd.DataFrame()
        for _df in data_list:
            data_df = pd.concat([data_df, pd.DataFrame.from_dict(_df, orient='columns')])
        data_df.to_csv(os.path.join(OutPath, f"Precision_{Target}_{Class}{suffix}_DATA.csv"), sep="\t")
    return True


def _make_precision_plot(df: pd.DataFrame, TestColumn: str = 'Y_true',
                         PredColumn: str = 'Y_pred',
                         RepeatColumn: str = 'Repeat',
                         FoldColumn: str = 'Fold',
                         DataSetColumn: str = 'Dataset',
                         OutPath: str = None,
                         n_thresholds: int = 50,
                         return_curve: bool = True,
                         plot_title="",
                         only_data:bool = False):
    if only_data:
        return_curve=True

    test_df = df[df[DataSetColumn] == 'test']
    repeats = test_df[RepeatColumn].unique()

    plt.figure(figsize=(10, 8))

    precisions = []
    thresholds = np.linspace(0, 1, n_thresholds)

    for r in repeats:
        repeat_df = test_df[test_df[RepeatColumn] == r]

        for f in repeat_df[FoldColumn].unique():
            fold_df = repeat_df[repeat_df[FoldColumn] == f]
            fold_precisions = []
            for threshold in thresholds:
                y_pred = (fold_df[PredColumn] >= threshold).astype(int)
                precision = precision_score(fold_df[TestColumn], y_pred, zero_division=1)
                fold_precisions.append(precision)
            if only_data == False:
                plt.plot(thresholds, fold_precisions, lw=1, alpha=0.1, color='black')
            precisions.append(fold_precisions)

    mean_precision = np.mean(precisions, axis=0)
    std_precision = np.std(precisions, axis=0)

    if only_data == False:
        plt.plot(thresholds, mean_precision, color='b',
             label=f'Mean Precision',
             lw=3, alpha=1)

    precision_upper = np.minimum(mean_precision + std_precision, 1)
    precision_lower = np.maximum(mean_precision - std_precision, 0)

    if only_data == False:
        plt.fill_between(thresholds, precision_lower, precision_upper, color='grey', alpha=.2)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Probability Threshold', size=20)
        plt.ylabel('Precision', size=20)
        plt.title(f'Precision vs Threshold. {plot_title}', size=20)
        plt.legend(loc="lower left", prop={'size': 20})

        if OutPath:
            plt.savefig(OutPath, dpi=300)
        else:
            plt.show()
    plt.close()

    if return_curve:
        return np.mean(mean_precision), np.mean(std_precision), (thresholds, mean_precision, std_precision)
    else:
        return np.mean(mean_precision), np.mean(std_precision), None


def make_f1_plots(df,
                  OutPath: str = None,
                  FoldColumn: str = 'Fold',
                  RepeatColumn: str = 'Repeat',
                  DataSetColumn: str = 'Dataset',
                  n_thresholds: int = 50,
                  Target: Literal=['Heart Axis', 'Muscle', 'Conduction'],
                  plot_title: str = "",
                  suffix: str = '',
                  mod_name: dict = {
                      'LR': 'Logistic Regression',
                      'XGB': 'eXtreme Gradient Boosting',
                      'customDT': 'Custom Decision Tree',
                      'normalDT': 'Normal Decision Tree',
                  },
                  only_data: bool=False
                  ):
    '''
    Function to create plots with proba_thresholds versus f1-score.
    '''
    pred_strings = [c for c in df.columns if c.startswith('Y_pred')]
    Classes = set([s.split("_")[3] for s in pred_strings])
    Models = set([s.split("_")[2] for s in pred_strings])

    for Class in Classes:
        data_list = []
        plt.figure(figsize=(10, 8))
        for Model in Models:
            _OutPath = os.path.join(OutPath, f"F1_curve_{Class}_{Model}{suffix}.svg")
            _Title = f"Target: {Target}, Class: {Class}, Model: {Model}"
            mean_f1, std_f1, f1_curve_data = _make_f1_plot(df,
                                                           TestColumn=f'Y_true_{Class}',
                                                           PredColumn=f'Y_pred_{Model}_{Class}{suffix}',
                                                           FoldColumn=FoldColumn,
                                                           RepeatColumn=RepeatColumn,
                                                           DataSetColumn=DataSetColumn,
                                                           OutPath=_OutPath,
                                                           n_thresholds=n_thresholds,
                                                           plot_title=_Title,
                                                           return_curve=True,
                                                           only_data = only_data)
            data_list.append({'thresholds': list(f1_curve_data[0]),
                              'mean_value': list(f1_curve_data[1]),
                              'std_value': list(f1_curve_data[2]),
                              'Model': len(f1_curve_data[0])*[Model]
                              })
            if only_data == False:
                plt.plot(f1_curve_data[0], f1_curve_data[1],
                         label=f"{mod_name[Model]}. Mean F1={round(mean_f1, 2)} ± {round(std_f1, 2)}", lw=2)
                line_color = plt.gca().lines[-1].get_color()
                f1_lower = np.maximum(f1_curve_data[1] - f1_curve_data[2], 0)
                f1_upper = np.minimum(f1_curve_data[1] + f1_curve_data[2], 1)
                plt.fill_between(f1_curve_data[0], f1_lower, f1_upper, color=line_color, alpha=.1)

        if only_data == False:
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('Probability Threshold', size=20)
            plt.ylabel('F1 Score', size=20)
            plt.title(f'F1 Score vs Threshold. Target: {Target}, Class: {Class}. {plot_title}', size=20)
            plt.legend(loc="lower left", prop={'size': 20})
            plt.savefig(os.path.join(OutPath, f"F1_{Target}_{Class}{suffix}.svg"), dpi=300)
        plt.close()

        # process data into dataframe and write to tsv file
        data_df = pd.DataFrame()
        for _df in data_list:
            data_df = pd.concat([data_df, pd.DataFrame.from_dict(_df, orient='columns')])
        data_df.to_csv(os.path.join(OutPath, f"F1_{Target}_{Class}{suffix}_DATA.csv"), sep="\t")

    return True


def _make_f1_plot(df: pd.DataFrame, TestColumn: str = 'Y_true',
                  PredColumn: str = 'Y_pred',
                  RepeatColumn: str = 'Repeat',
                  FoldColumn: str = 'Fold',
                  DataSetColumn: str = 'Dataset',
                  OutPath: str = None,
                  n_thresholds: int = 50,
                  return_curve: bool = True,
                  plot_title: str = "",
                  only_data: bool = False):

    if only_data:
        return_curve=True

    test_df = df[df[DataSetColumn] == 'test']
    repeats = test_df[RepeatColumn].unique()

    plt.figure(figsize=(10, 8))

    f1_scores = []
    thresholds = np.linspace(0, 1, n_thresholds)

    for r in repeats:
        repeat_df = test_df[test_df[RepeatColumn] == r]

        for f in repeat_df[FoldColumn].unique():
            fold_df = repeat_df[repeat_df[FoldColumn] == f]
            fold_f1_scores = []
            for threshold in thresholds:
                y_pred = (fold_df[PredColumn] >= threshold).astype(int)
                f1 = f1_score(fold_df[TestColumn], y_pred)
                fold_f1_scores.append(f1)

            if only_data == False:
                plt.plot(thresholds, fold_f1_scores, lw=1, alpha=0.1, color='black')
            f1_scores.append(fold_f1_scores)

    mean_f1 = np.mean(f1_scores, axis=0)
    std_f1 = np.std(f1_scores, axis=0)

    if only_data == False:
        plt.plot(thresholds, mean_f1, color='b',
                 label=f'Mean F1 Score',
                 lw=3, alpha=1)
        f1_upper = np.minimum(mean_f1 + std_f1, 1)
        f1_lower = np.maximum(mean_f1 - std_f1, 0)
        plt.fill_between(thresholds, f1_lower, f1_upper, color='grey', alpha=.2)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Probability Threshold', size=20)
        plt.ylabel('F1 Score', size=20)
        plt.title(f'F1 Score vs Threshold. {plot_title}', size=20)
        plt.legend(loc="lower left", prop={'size': 20})

        if OutPath:
            plt.savefig(OutPath, dpi=300)
        else:
            plt.show()
    plt.close()

    if return_curve:
        return np.mean(mean_f1), np.mean(std_f1), (thresholds, mean_f1, std_f1)
    else:
        return np.mean(mean_f1), np.mean(std_f1), None