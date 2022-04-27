import numpy as np
import string
import re
from collections import Counter
from sklearn.metrics import matthews_corrcoef, f1_score
from scipy.stats import pearsonr, spearmanr

from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from ..config import METRICS


def accuracy(prediction, ground_truth):
    return prediction.lower() == ground_truth.lower()


def evaluate_func(predictions, data, metric, return_all=False):

    def cast_to_float(predictions):
        new_predictions = []
        for prediction in predictions:
            try:
                new_predictions.append(float(prediction.strip()))
            except:
                new_predictions.append(float('NaN'))
        assert len(new_predictions) == len(predictions)
        return new_predictions

    assert len(predictions) == len(data)

    all_metrics = [m.strip() for m in metric.split("|")]
    results = {}
    results_all = {}
    for m in all_metrics:
        if m == "EM":
            ems = []
            for (prediction, dp) in zip(predictions, data):
                ems.append(get_exact_match_over_list(prediction, dp[1]))
            results[m] = np.mean(ems)
            results_all[m] = [bool(_i) for _i in ems]
        elif m == "SoftEM":
            ems = []
            for (prediction, dp) in zip(predictions, data):
                ems.append(get_soft_exact_match_over_list(prediction, dp[1]))
            results[m] = np.mean(ems)
            results_all[m] = [bool(_i) for _i in ems]
        elif m == "ACC":
            accs = []
            for (prediction, dp) in zip(predictions, data):
                accs.append(get_accuracy_over_list(prediction, dp[1]))
            results[m] = np.mean(accs)
            results_all[m] = accs
        elif m == 'BLEU':
            scores = [get_bleu_score(pred, truth[1]) for pred, truth in zip(predictions, data)]
            results[m] = np.mean(scores)
            results_all[m] = scores
        elif m == "QA-F1":
            f1s = []
            for (prediction, dp) in zip(predictions, data):
                f1s.append(get_f1_over_list(prediction, dp[1]))
            results[m] = np.mean(f1s)
            # results_all[m] = f1s
            results_all[m] = [float(_i) for _i in f1s]
        elif m == "Classification-F1":
            results[m] = f1_score([dp[1][0] for dp in data], predictions, average="macro")
        elif m == "Matthew-Correlation":
            results[m] = get_matthews_corr(data, predictions)
        elif m == "Pearson-Correlation":
            predictions = cast_to_float(predictions)
            results[m] = pearsonr([float(dp[1][0]) for dp in data], predictions)[0]
        elif m == "Rouge-L":
            rouges = []
            for (prediction, dp) in zip(predictions, data):
                rouges.append(get_rouge_over_list(prediction, dp[1]))
            results[m] = np.mean(rouges)
        else:
            # Unsupported metric, use EM instead
            ems = []
            for (prediction, dp) in zip(predictions, data):
                ems.append(get_exact_match_over_list(prediction, dp[1]))
            results[m] = np.mean(ems)
            results_all[m] = [bool(_i) for _i in ems]

    if return_all:
        return results, results_all
    return results


def get_bleu_score(prediction, ground_truths):
    prediction_tokens = prediction.split()
    ground_truth_tokens = [i.split() for i in ground_truths]
    return sentence_bleu(ground_truth_tokens, prediction_tokens)


def get_matthews_corr(data, predictions):
    # only cola is using this...?
    new_predictions = []
    for prediction in predictions:
        if prediction.strip() == "acceptable":
            new_predictions.append(1.0)
        else:
            new_predictions.append(0.0)
    new_gold = []
    for dp in data:
        if dp[1][0] == "acceptable":
            new_gold.append(1.0)
        else:
            new_gold.append(0.0)
    return matthews_corrcoef(new_gold, new_predictions)


def qa_f1_score(prediction, ground_truth):
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_rouge_over_list(prediction, groundtruth):

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    if len(remove_punc(prediction)) == 0:
        return 0.0  # during early stages, it might generate nothin?
    # print(prediction)
    rouge = Rouge()
    if type(groundtruth) == list:
        if len(groundtruth) == 0:
            return 0
        return np.max(
            [rouge.get_scores(prediction, gt, avg=True)["rouge-l"]["f"] for gt in groundtruth])
    return rouge.get_scores(prediction, groundtruth, avg=True)["rouge-l"]["f"]


def get_accuracy_over_list(prediction, groundtruth):
    if type(groundtruth) == list:
        if len(groundtruth) == 0:
            return 0
        return np.max([accuracy(prediction, gt) for gt in groundtruth])
    return accuracy(prediction, groundtruth)


def get_f1_over_list(prediction, groundtruth):
    # if type(groundtruth)==list:
    if len(groundtruth) == 0:
        return 0
    prediction_norm = normalize_answer(prediction)
    return np.max([qa_f1_score(prediction_norm, normalize_answer(gt)) for gt in groundtruth])
    # return qa_f1_score(prediction, groundtruth)


def get_exact_match_over_list(prediction, groundtruth):
    # if type(groundtruth)==list:
    if len(groundtruth) == 0:
        return 0
    prediction_norm = normalize_answer(prediction)
    return np.max([(prediction_norm == normalize_answer(gt)) for gt in groundtruth])

    # return (normalize_answer(prediction) == groundtruth)

def get_soft_exact_match_over_list(prediction, groundtruth):
    if len(groundtruth) == 0:
        return 0
    prediction_norm = normalize_answer(prediction)
    def soft_match(a, b):
        return (len(a) and len(b)) and (a in b or b in a or a == b)
    return np.max([soft_match(prediction_norm, normalize_answer(gt)) for gt in groundtruth])

    # return (normalize_answer(prediction) == groundtruth)

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
