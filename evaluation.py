import numpy as np
from collections import Counter
from scipy.optimize import linear_sum_assignment


# CoNLL-2012 metrics https://aclanthology.org/W12-4501.pdf#page=19
class OfficialCoNLL2012CorefEvaluator(object):
    def __init__(self):
        # muc/b_cubed/ceafe make up the official conll2012 average (unweighted mean
        # of these three, per the shared task definition). lea is tracked alongside
        # them but is intentionally excluded from that average, since it isn't part
        # of the official CoNLL-2012 score.
        self.evaluators = [OfficialEvaluator(m) for m in (muc, b_cubed, ceafe)]
        self.metrics = {"muc": 0, "b_cubed": 1, "ceafe": 2}
        self.lea_evaluator = OfficialEvaluator(lea)

    def update(self, pred, gold, mention_to_predicted, mention_to_gold):
        for e in self.evaluators:
            e.update(pred, gold, mention_to_predicted, mention_to_gold)
        self.lea_evaluator.update(pred, gold, mention_to_predicted, mention_to_gold)

    def _aggregate(self, metric, getter):
        if metric == "lea":
            return getter(self.lea_evaluator)
        if metric == "conll2012":
            return sum(getter(e) for e in self.evaluators) / len(self.evaluators)
        return getter(self.evaluators[self.metrics[metric]])

    def get_f1(self, metric):
        return self._aggregate(metric, OfficialEvaluator.get_f1)

    def get_recall(self, metric):
        return self._aggregate(metric, OfficialEvaluator.get_recall)

    def get_precision(self, metric):
        return self._aggregate(metric, OfficialEvaluator.get_precision)

    def get_prf(self, metric):
        return self.get_precision(metric), self.get_recall(metric), self.get_f1(metric)


class OfficialEvaluator(object):
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        if self.metric == ceafe:
            pn, pd, rn, rd = self.metric(predicted, gold)
        else:
            pn, pd = self.metric(predicted, mention_to_gold)
            rn, rd = self.metric(gold, mention_to_predicted)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den


def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)


# mention based metric (Bagga and Baldwin, 1998)
def b_cubed(clusters, mention_to_gold_cluster):
    num, den = 0.0, 0

    for cluster in clusters:
        gold_counts = Counter()

        for mention in cluster:
            if mention in mention_to_gold_cluster:
                gold_counts[mention_to_gold_cluster[mention]] += 1

        correct = sum(count * count for count in gold_counts.values())

        if cluster:
            num += correct / len(cluster)
            den += len(cluster)

    return num, den


# link based metric (Vilain et al. 1995).
def muc(clusters, mention_to_gold_cluster):
    tp, p = 0, 0

    for cluster in clusters:
        p += len(cluster) - 1
        tp += len(cluster)

        linked = set()

        for mention in cluster:
            if mention in mention_to_gold_cluster:
                linked.add(mention_to_gold_cluster[mention])
            else:
                tp -= 1

        tp -= len(linked)

    return tp, p


# Entity based metric (Luo, 2005)
def ceafe(clusters, gold_clusters):
    if not clusters or not gold_clusters:
        return 0.0, len(clusters), 0.0, len(gold_clusters)

    gold_index = {
        mention: i
        for i, cluster in enumerate(gold_clusters)
        for mention in cluster
    }

    scores = np.zeros((len(gold_clusters), len(clusters)))

    for pred_idx, pred_cluster in enumerate(clusters):
        overlap = Counter(
            gold_index[mention]
            for mention in pred_cluster
            if mention in gold_index
        )

        for gold_idx, count in overlap.items():
            gold_cluster = gold_clusters[gold_idx]
            scores[gold_idx, pred_idx] = (
                2.0 * count / (len(gold_cluster) + len(pred_cluster))
            )

    row_ind, col_ind = linear_sum_assignment(-scores)
    similarity = float(scores[row_ind, col_ind].sum())

    return (
        similarity,
        len(clusters),
        similarity,
        len(gold_clusters),
    )


def lea(clusters, mention_to_gold):
    num, dem = 0.0, 0

    for c in clusters:
        size = len(c)
        if size < 2:
            continue

        counts = Counter(mention_to_gold[m] for m in c if m in mention_to_gold)
        common_links = sum(n * (n - 1) // 2 for n in counts.values())
        all_links = size * (size - 1) / 2

        num += size * common_links / all_links
        dem += size

    return num, dem


def _to_tuple_clusters(clusters):
    """Convert [[[0], [4], [5]], [[1], [9]]] -> [[(0,), (4,), (5,)], [(1,), (9,)]]

    Mentions need to be hashable (tuples) so they can be used as dict keys
    and set members. Works for single-token mentions ([0]) or multi-token
    spans ([0, 3]) alike.
    """
    return [[tuple(mention) for mention in cluster] for cluster in clusters]


def _mention_to_cluster_map(clusters):
    """Build {mention: cluster_index} for every mention in every cluster."""
    mapping = {}
    for i, cluster in enumerate(clusters):
        for m in cluster:
            if m in mapping:
                raise ValueError(f"mention {m} appears more than once")
            mapping[m] = i
    return mapping


def get_coref_metrics(gold_clusters, predicted_clusters):
    """gold_clusters / predicted_clusters: e.g. gold["clusters"], predicted["clusters"]
        [[[0], [4], [5], [6]], [[1], [9], [10]], ...]

    Returns a dict {"MUC": {"P":..,"R":..,"F1":..}, "B3": {...}, "CEAF_e": {...}, "conll": {...}}
    """
    gold = _to_tuple_clusters(gold_clusters)
    pred = _to_tuple_clusters(predicted_clusters)

    mention_to_gold = _mention_to_cluster_map(gold)
    mention_to_pred = _mention_to_cluster_map(pred)

    evaluator = OfficialCoNLL2012CorefEvaluator()
    evaluator.update(pred, gold, mention_to_pred, mention_to_gold)

    result = {}
    name_map = {"muc": "MUC", "b_cubed": "B3", "ceafe": "CEAF_e"}
    for key, label in name_map.items():
        p, r, f = evaluator.get_prf(key)
        result[label] = {"P": p, "R": r, "F1": f}

    # LEA is reported separately: it's not part of the official conll2012
    # average, and it skips singleton clusters by design (link-based metric).
    p, r, f = evaluator.get_prf("lea")
    result["LEA"] = {"P": p, "R": r, "F1": f}

    p, r, f = evaluator.get_prf("conll2012")
    result["conll"] = {"P": p, "R": r, "F1": f}

    return result
