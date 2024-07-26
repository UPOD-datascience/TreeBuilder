import numpy as np
from scipy import stats
import pandas as pd
import pickle
import os
import tqdm

from typing import List, Callable

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn.calibration import calibration_curve

#TODO: add rulefitter
#TODO: add net benefit plotter
#TODO: calibration curve plotter

def rulefitter():
    # RuleFit: https://github.com/christophM/rulefit
    # Imodels: https://github.com/csinva/imodels
    # Skope-rules: https://github.com/scikit-learn-contrib/skope-rules
    pass


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
            nb_thresholds, net_benefits, all_positive, all_negative = net_benefit_curve(y_true, y_pred_proba, thresholds)

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
        plt.title(f'Calibration Curves; {plot_title}; {target}')
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

    le_pipe_rf = PipeDict['rf']
    le_pipe_gbc = PipeDict['gbc']
    le_pipe_xgb = PipeDict['xgb']
    le_pipe_dt = PipeDict['dt']

    for i, (train_index, test_index) in tqdm.tqdm(enumerate(Splitter.split(X, Y)), total=num_splits * num_repeats):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        if use_class_weights == True:
            WeightMap = dict(zip(np.unique(Y_train), np.bincount(Y_train)))
            VectorizedMapping = np.vectorize(WeightMap.get)
            class_counts = 1000 / VectorizedMapping(Y_train)
        else:
            class_counts = None
        # Train models
        le_pipe_rf.fit(X_train, Y_train, RandomForest__sample_weight=class_counts)
        le_pipe_gbc.fit(X_train, Y_train, GradientBoosting__sample_weight=class_counts)
        le_pipe_xgb.fit(X_train, Y_train, XGBoost__sample_weight=class_counts)
        le_pipe_dt.fit(X_train, Y_train, DecisionTree__sample_weight=class_counts)

        # Predict on test set
        Y_pred_rf = le_pipe_rf.predict_proba(X_test)
        Y_pred_gbc = le_pipe_gbc.predict_proba(X_test)
        Y_pred_xgb = le_pipe_xgb.predict_proba(X_test)
        Y_pred_dt = le_pipe_dt.predict_proba(X_test)

        res_dict = {
                'Y_test': Y_test,
                'Y_pred_rf': Y_pred_rf,
                'Y_pred_gbc': Y_pred_gbc,
                'Y_pred_xgb': Y_pred_xgb,
                'Y_pred_dt': Y_pred_dt,
                'Fold': i % num_splits,
                'Repeat': i // num_splits,
            }
        RES_LIST.append(
            res_dict.copy()
        )
        if make_df:
            #if num_classes==2:
            #    _df = pd.DataFrame(res_dict)
            #else:
            res_dict['Fold'] = len(Y_test) * [res_dict['Fold']]
            res_dict['Repeat'] = len(Y_test) * [res_dict['Repeat']]

            for i in range(num_classes):
                res_dict[f'Y_test_{ClassMap[i]}'] = (Y_test==i).astype(int)
                for mod in ['rf', 'gbc', 'xgb', 'dt']:
                    res_dict[f'Y_pred_{mod}_{ClassMap[i]}'] \
                        = res_dict[f'Y_pred_{mod}'][:, i]
            for mod in ['rf', 'gbc', 'xgb', 'dt']:
                res_dict.pop(f'Y_pred_{mod}')
            df_list.append(pd.DataFrame(res_dict))

    if make_df:
        df = pd.concat(df_list, ignore_index=True)
    else:
        df = None
    return RES_LIST, df


def make_plots(results,
               labelcoder,
               n_classes,
               colors,
               ClassMap,
               output_map=None,
               show_plot=False,
               plot_title=""):
    assert output_map is not None, "output_map is required"
    os.makedirs(output_map, exist_ok=True)
    plots = []

    for i in range(n_classes):
        fig, ax = plt.subplots()
        mean_fpr = np.linspace(0, 1, 100)

        tprs_rf = []
        tprs_gbc = []
        tprs_xgb = []
        tprs_dt = []
        aucs_rf = []
        aucs_gbc = []
        aucs_xgb = []
        aucs_dt = []

        for res in results:
            y_test_bin = labelcoder.transform(res['Y_test'])
            if y_test_bin.shape[1]==1:
                ip = 1 # if the pred only has one value then by default we assume it represents the positive case
            else:
                ip = i
            fpr_rf, tpr_rf, _ = roc_curve(y_test_bin[:, i], res['Y_pred_rf'][:, ip])
            fpr_gbc, tpr_gbc, _ = roc_curve(y_test_bin[:, i], res['Y_pred_gbc'][:, ip])
            fpr_xgb, tpr_xgb, _ = roc_curve(y_test_bin[:, i], res['Y_pred_xgb'][:, ip])
            fpr_dt, tpr_dt, _ = roc_curve(y_test_bin[:, i], res['Y_pred_dt'][:, ip])

            interp_tpr_rf = np.interp(mean_fpr, fpr_rf, tpr_rf)
            interp_tpr_gbc = np.interp(mean_fpr, fpr_gbc, tpr_gbc)
            interp_tpr_xgb = np.interp(mean_fpr, fpr_xgb, tpr_xgb)
            interp_tpr_dt = np.interp(mean_fpr, fpr_dt, tpr_dt)

            interp_tpr_rf[0] = 0.0
            interp_tpr_gbc[0] = 0.0
            interp_tpr_xgb[0] = 0.0
            interp_tpr_dt[0] = 0.0

            tprs_rf.append(interp_tpr_rf)
            tprs_gbc.append(interp_tpr_gbc)
            tprs_xgb.append(interp_tpr_xgb)
            tprs_dt.append(interp_tpr_dt)

            aucs_rf.append(auc(fpr_rf, tpr_rf))
            aucs_gbc.append(auc(fpr_gbc, tpr_gbc))
            aucs_xgb.append(auc(fpr_xgb, tpr_xgb))
            aucs_dt.append(auc(fpr_dt, tpr_dt))

        mean_tpr_rf = np.mean(tprs_rf, axis=0)
        mean_tpr_gbc = np.mean(tprs_gbc, axis=0)
        mean_tpr_xgb = np.mean(tprs_xgb, axis=0)
        mean_tpr_dt = np.mean(tprs_dt, axis=0)

        mean_tpr_rf[-1] = 1.0
        mean_tpr_gbc[-1] = 1.0
        mean_tpr_xgb[-1] = 1.0
        mean_tpr_dt[-1] = 1.0

        mean_auc_rf = auc(mean_fpr, mean_tpr_rf)
        mean_auc_gbc = auc(mean_fpr, mean_tpr_gbc)
        mean_auc_xgb = auc(mean_fpr, mean_tpr_xgb)
        mean_auc_dt = auc(mean_fpr, mean_tpr_dt)

        std_tpr_rf = np.std(tprs_rf, axis=0)
        std_tpr_gbc = np.std(tprs_gbc, axis=0)
        std_tpr_xgb = np.std(tprs_xgb, axis=0)
        std_tpr_dt = np.std(tprs_dt, axis=0)

        # 95% confidence interval
        ci_tpr_rf = stats.norm.ppf(0.975) * std_tpr_rf / np.sqrt(len(tprs_rf))
        ci_tpr_gbc = stats.norm.ppf(0.975) * std_tpr_gbc / np.sqrt(len(tprs_gbc))
        ci_tpr_xgb = stats.norm.ppf(0.975) * std_tpr_xgb / np.sqrt(len(tprs_xgb))
        ci_tpr_dt = stats.norm.ppf(0.975) * std_tpr_dt / np.sqrt(len(tprs_dt))

        ax.plot(mean_fpr, mean_tpr_rf, color=colors[0], lw=2, label=f'Random Forest (mean AUC = {mean_auc_rf:0.2f})')
        ax.fill_between(mean_fpr, mean_tpr_rf - ci_tpr_rf, mean_tpr_rf + ci_tpr_rf, color=colors[0], alpha=0.2)

        ax.plot(mean_fpr, mean_tpr_gbc, color=colors[1], lw=2,
                label=f'Gradient Boosting (mean AUC = {mean_auc_gbc:0.2f})')
        ax.fill_between(mean_fpr, mean_tpr_gbc - ci_tpr_gbc, mean_tpr_gbc + ci_tpr_gbc, color=colors[1], alpha=0.2)

        ax.plot(mean_fpr, mean_tpr_xgb, color=colors[1], lw=2, label=f'XGBoosting (mean AUC = {mean_auc_xgb:0.2f})')
        ax.fill_between(mean_fpr, mean_tpr_xgb - ci_tpr_xgb, mean_tpr_xgb + ci_tpr_xgb, color=colors[1], alpha=0.2)

        ax.plot(mean_fpr, mean_tpr_dt, color=colors[2], lw=2, label=f'Decision Tree (mean AUC = {mean_auc_dt:0.2f})')
        ax.fill_between(mean_fpr, mean_tpr_dt - ci_tpr_dt, mean_tpr_dt + ci_tpr_dt, color=colors[2], alpha=0.2)

        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        if n_classes == 2:
            class_string = f" {ClassMap[0]}/{ClassMap[1]}"
        else:
            class_string = f"Class {ClassMap[i]}"
        ax.set_title(f'ROC Curve; {plot_title}; {class_string}')
        ax.legend(loc="lower right")

        if show_plot:
            plt.show()

        # Save the figure to the output directory
        filename = os.path.join(output_map, f'{ClassMap[i]}_ROC.png')
        fig.savefig(filename, dpi=300)
        plt.close(fig)  # Close the figure to free up memory

        plots.append(fig)
        if n_classes == 2:
            return plots
    return plots


def get_performance(results: List = None, threshold: float = 0.5, ClassMap: dict = None, binarizer: Callable = None):
    perf_list = []
    n_classes = len(ClassMap)
    SPEC = lambda tn, fp: tn / (tn + fp)
    for res in results:
        Y_test_bin = binarizer.transform(res['Y_test'])
        Y_pred_rf = res['Y_pred_rf']
        Y_pred_gbt = res['Y_pred_gbc']
        Y_pred_xgb = res['Y_pred_xgb']
        Y_pred_dt = res['Y_pred_dt']

        for i in range(n_classes):
            if n_classes == 2:
                class_string = f"{ClassMap[0]}/{ClassMap[1]}"
            else:
                class_string = ClassMap[i]

            if Y_test_bin.shape[1] == 1:
                ip = 1  # if the pred only has one value then by default we assume it represents the positive case
            else:
                ip = i

            cm = confusion_matrix(Y_test_bin[:, i], np.where(Y_pred_rf[:, ip] > threshold, 1, 0))
            tn = cm[0, 0]
            tp = cm[1, 1]
            fn = cm[1, 0]
            fp = cm[0, 1]
            res_dict = {
                'f1': f1_score(Y_test_bin[:, i], np.where(Y_pred_rf[:, ip] > threshold, 1, 0)),
                'precision': precision_score(Y_test_bin[:, i], np.where(Y_pred_rf[:, ip] > threshold, 1, 0)),
                'recall': recall_score(Y_test_bin[:, i], np.where(Y_pred_rf[:, ip] > threshold, 1, 0)),
                'specificity': SPEC(fn, fp),
                'tn': tn,
                'tp': tp,
                'fn': fn,
                'fp': fp,
                'model': 'RF',
                'Class': class_string,
                'Fold': res['Fold'],
                'Repeat': res['Repeat']
            }
            perf_list.append(res_dict)

            cm = confusion_matrix(Y_test_bin[:, i], np.where(Y_pred_gbt[:, ip] > threshold, 1, 0))
            tn = cm[0, 0]
            tp = cm[1, 1]
            fn = cm[1, 0]
            fp = cm[0, 1]
            res_dict = {
                'f1': f1_score(Y_test_bin[:, i], np.where(Y_pred_gbt[:, ip] > threshold, 1, 0)),
                'precision': precision_score(Y_test_bin[:, i], np.where(Y_pred_gbt[:, ip] > threshold, 1, 0)),
                'recall': recall_score(Y_test_bin[:, i], np.where(Y_pred_gbt[:, ip] > threshold, 1, 0)),
                'specificity': SPEC(fn, fp),
                'tn': tn,
                'tp': tp,
                'fn': fn,
                'fp': fp,
                'model': 'GBT',
                'Class': class_string,
                'Fold': res['Fold'],
                'Repeat': res['Repeat']
            }
            perf_list.append(res_dict)

            cm = confusion_matrix(Y_test_bin[:, i], np.where(Y_pred_xgb[:, ip] > threshold, 1, 0))
            tn = cm[0, 0]
            tp = cm[1, 1]
            fn = cm[1, 0]
            fp = cm[0, 1]
            res_dict = {
                'f1': f1_score(Y_test_bin[:, i], np.where(Y_pred_xgb[:, ip] > threshold, 1, 0)),
                'precision': precision_score(Y_test_bin[:, i], np.where(Y_pred_xgb[:, ip] > threshold, 1, 0)),
                'recall': recall_score(Y_test_bin[:, i], np.where(Y_pred_xgb[:, ip] > threshold, 1, 0)),
                'specificity': SPEC(fn, fp),
                'tn': tn,
                'tp': tp,
                'fn': fn,
                'fp': fp,
                'model': 'XGB',
                'Class': class_string,
                'Fold': res['Fold'],
                'Repeat': res['Repeat']
            }
            perf_list.append(res_dict)

            cm = confusion_matrix(Y_test_bin[:, i], np.where(Y_pred_rf[:, ip] > threshold, 1, 0))
            tn = cm[0, 0]
            tp = cm[1, 1]
            fn = cm[1, 0]
            fp = cm[0, 1]
            res_dict = {
                'f1': f1_score(Y_test_bin[:, i], np.where(Y_pred_dt[:, ip] > threshold, 1, 0)),
                'precision': precision_score(Y_test_bin[:, i], np.where(Y_pred_dt[:, ip] > threshold, 1, 0)),
                'recall': recall_score(Y_test_bin[:, i], np.where(Y_pred_dt[:, ip] > threshold, 1, 0)),
                'specificity': SPEC(fn, fp),
                'tn': tn,
                'tp': tp,
                'fn': fn,
                'fp': fp,
                'model': 'DT',
                'Class': ClassMap[i],
                'Fold': res['Fold'],
                'Repeat': res['Repeat']
            }
            perf_list.append(res_dict)
            if n_classes == 2:
                break
    return perf_list