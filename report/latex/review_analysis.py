"""Paired image bootstrap, prediction concentration and readable plots from audited results."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common import digest

LEVELS = ['mild', 'medium', 'severe']
LABELS = ['Nhẹ', 'Vừa', 'Nặng']
PRE, POST = 'pre__gaussian__clahe', 'post__gaussian__clahe'
METHODS = ['identity', 'clahe', 'diffusion', 'pre__gaussian__gamma']
NAMES = ['Identity', 'CLAHE', 'Diffusion', 'Gaussian → Gamma']


def macro_scores(labels, predictions, weights):
    """One Macro-F1 per fixed run, with multiplicity weights on source images."""
    support = np.bincount(labels, weights=weights, minlength=43)
    result = []
    for pred in predictions:
        predicted = np.bincount(pred, weights=weights, minlength=43)
        correct = pred == labels
        tp = np.bincount(labels[correct], weights=weights[correct], minlength=43)
        denominator = support + predicted
        result.append(np.divide(2 * tp, denominator, out=np.zeros(43), where=denominator > 0).mean())
    return np.asarray(result)


def paired_bootstrap(labels, before, after, repetitions=2000, seed=20261025, checkpoint_groups=None):
    """Stratify by true class; identical image weights for both methods and all runs."""
    labels = np.asarray(labels, dtype=int)
    before, after = np.asarray(before, dtype=int), np.asarray(after, dtype=int)
    if (labels.ndim != 1 or not len(labels) or before.shape != after.shape
            or before.ndim != 2 or before.shape[1] != len(labels) or before.shape[0] < 1
            or repetitions < 1 or any(a.min() < 0 or a.max() >= 43 for a in [labels, before, after])):
        raise ValueError('Expected aligned nonempty runs/images and labels 0..42')
    classes = [np.flatnonzero(labels == c) for c in range(43)]
    classes = [indices for indices in classes if len(indices)]
    weights = np.ones(len(labels))
    point = 100 * (macro_scores(labels, after, weights) - macro_scores(labels, before, weights)).mean()
    rng = np.random.default_rng(seed)
    group_indices = None
    group_rng = np.random.default_rng(seed + 1)
    if checkpoint_groups is not None:
        groups = np.asarray(checkpoint_groups)
        if groups.shape != (before.shape[0],):
            raise ValueError('One checkpoint group is required per fixed run')
        group_indices = [np.flatnonzero(groups == value) for value in np.unique(groups)]
    draws = np.empty(repetitions)
    for b in range(repetitions):
        for indices in classes:
            weights[indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)))
        differences = 100 * (macro_scores(labels, after, weights) - macro_scores(labels, before, weights))
        if group_indices is None:
            draws[b] = differences.mean()
        else:
            by_checkpoint = np.array([differences[idx].mean() for idx in group_indices])
            draws[b] = by_checkpoint[group_rng.integers(len(by_checkpoint), size=len(by_checkpoint))].mean()
    return float(point), draws


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261025)
    args = parser.parse_args()
    evidence = ROOT / 'report/evidence/final'
    audit = json.loads((evidence / 'audit.json').read_text(encoding='utf-8'))
    if audit['archive_sha256'] != digest(args.archive):
        raise ValueError('Use the same results ZIP as the completed audit')
    images = ROOT / 'report/latex/images'
    tables = ROOT / 'report/latex/tables'
    cache = ROOT / 'data/processed/report-review'
    cache.mkdir(parents=True, exist_ok=True)
    compact = cache / 'order_predictions.npz'
    run_keys = [(t, n) for t in [11, 22, 33] for n in [101, 202, 303]]
    with zipfile.ZipFile(args.archive) as z:
        checksums = json.loads(z.read('SHA256.json'))
        def verified(name):
            data = z.read(name)
            if hashlib.sha256(data).hexdigest() != checksums[name]:
                raise ValueError('Archive member changed: ' + name)
            return data
        protocol = json.loads(verified('08_processing_extension/fulltest/protocol.json'))
        summary = list(csv.DictReader(io.StringIO(verified('08_processing_extension/fulltest/summary.csv').decode('utf-8'))))
        with np.load(io.BytesIO(verified('08_processing_extension/fulltest/confusion_matrices.npz')), allow_pickle=False) as npz:
            matrices = dict(zip(npz['run_ids'].tolist(), npz['matrices']))
        ids = protocol['sample_ids']
        if len(ids) != 12630 or len(set(ids)) != len(ids):
            raise ValueError('Expected the complete unique official test set')
        indices = {sid: i for i, sid in enumerate(ids)}
        if compact.exists():
            with np.load(compact, allow_pickle=False) as npz:
                if (str(npz['archive_sha256']) != audit['archive_sha256']
                        or npz['sample_ids'].tolist() != ids):
                    raise ValueError('Stale prediction cache')
                labels, predictions = npz['labels'], npz['predictions']
        else:
            labels = np.full(len(ids), 255, dtype=np.uint8)
            predictions = np.full((3, 2, 9, len(ids)), 255, dtype=np.uint8)
            with z.open('08_processing_extension/fulltest/predictions.csv') as binary:
                for row in csv.DictReader(io.TextIOWrapper(binary, encoding='utf-8', newline='')):
                    if row['method'] not in [PRE, POST] or row['condition'] not in LEVELS:
                        continue
                    i = indices[row['sample_id']]
                    key = (LEVELS.index(row['condition']), [PRE, POST].index(row['method']),
                           run_keys.index((int(row['train_seed']), int(row['noise_seed']))), i)
                    if predictions[key] != 255:
                        raise ValueError('Duplicate paired prediction')
                    label = int(row['class_id'])
                    if labels[i] not in [255, label]:
                        raise ValueError('Inconsistent label across paired runs')
                    labels[i] = label
                    predictions[key] = int(row['prediction'])
            if labels.max() >= 43 or predictions.max() >= 43:
                raise ValueError('Missing or invalid predictions')
            np.savez_compressed(compact, archive_sha256=audit['archive_sha256'], sample_ids=np.array(ids),
                                labels=labels, predictions=predictions)
            print('Aligned 682,020 paired prediction rows.', flush=True)
        support = np.bincount(labels, minlength=43)
        # Match every cached run to the independently audited confusion matrix.
        for c, condition in enumerate(LEVELS):
            for m, method in enumerate([PRE, POST]):
                for r, (train, noise) in enumerate(run_keys):
                    recorded = next(s for s in summary if s['condition'] == condition and s['method'] == method
                                    and int(s['train_seed']) == train and int(s['noise_seed']) == noise)
                    actual = np.bincount(43 * labels.astype(int) + predictions[c, m, r], minlength=43 * 43).reshape(43, 43)
                    if not np.array_equal(actual, matrices[recorded['run_id']]):
                        raise ValueError('Paired run disagrees with audited matrix')
        results = []
        draws_out = {}
        for c, condition in enumerate(LEVELS):
            point, draws = paired_bootstrap(labels, predictions[c, 0], predictions[c, 1], args.repetitions, args.seed)
            ci = np.quantile(draws, [.025, .975])
            _, hierarchical = paired_bootstrap(labels, predictions[c, 0], predictions[c, 1], args.repetitions, args.seed,
                                                checkpoint_groups=[t for t, n in run_keys])
            hci = np.quantile(hierarchical, [.025, .975])
            expected = []
            for method in [PRE, POST]:
                part = [r for r in summary if r['condition'] == condition and r['method'] == method]
                expected.append(np.mean([float(r['macro_f1']) for r in part]))
            if abs(point - 100 * (expected[1] - expected[0])) > 1e-10:
                raise ValueError('Bootstrap point differs from the report aggregation')
            results.append(dict(condition=condition, delta_macro_f1_pp=point, lower_95_pp=float(ci[0]),
                                upper_95_pp=float(ci[1]), checkpoint_image_lower_95_pp=float(hci[0]),
                                checkpoint_image_upper_95_pp=float(hci[1]), repetitions=args.repetitions, seed=args.seed))
            draws_out[condition] = draws
            draws_out[condition + '_checkpoint_image'] = hierarchical
            print('Paired bootstrap', condition, results[-1], flush=True)
        with (evidence / 'order_bootstrap.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
        (tables / 'order_bootstrap.tex').write_text('\n'.join(
            f"{LABELS[i]} & {r['delta_macro_f1_pp']:+.3f} & [{r['lower_95_pp']:+.3f}; {r['upper_95_pp']:+.3f}] & "
            f"[{r['checkpoint_image_lower_95_pp']:+.3f}; {r['checkpoint_image_upper_95_pp']:+.3f}] " + r'\\'
            for i, r in enumerate(results)) + '\n', encoding='utf-8')
        np.savez_compressed(cache / 'bootstrap_draws.npz', **draws_out)
        # Prediction frequency is averaged over runs; F1 is not computed from a pooled matrix here.
        concentration = []
        for condition in LEVELS:
            for method, name in zip(METHODS, NAMES):
                part = [r for r in summary if r['condition'] == condition and r['method'] == method]
                values = []
                for r in part:
                    matrix = matrices[r['run_id']]
                    if not np.array_equal(matrix.sum(1), support):
                        raise ValueError('True class supports disagree')
                    predicted_5, tp_5 = matrix[:, 5].sum(), matrix[5, 5]
                    values.append((100 * predicted_5 / len(ids), 100 * (predicted_5 - tp_5) / len(ids),
                                   100 * tp_5 / predicted_5 if predicted_5 else 0))
                mean = np.mean(values, axis=0)
                concentration.append(dict(condition=condition, method=method, true_class_5_count=int(support[5]),
                                          true_class_5_percent=100 * support[5] / len(ids),
                                          predicted_class_5_percent=float(mean[0]), false_class_5_percent=float(mean[1]),
                                          class_5_precision_percent=float(mean[2])))
        with (evidence / 'class5_concentration.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(concentration[0])); writer.writeheader(); writer.writerows(concentration)
        severe = [r for r in concentration if r['condition'] == 'severe']
        (tables / 'class5_concentration.tex').write_text('\n'.join(
            f"{name.replace(' → ', r' $\to$ ')} & {r['predicted_class_5_percent']:.2f} & {r['false_class_5_percent']:.2f} & {r['class_5_precision_percent']:.2f} " + r'\\'
            for name, r in zip(NAMES, severe)) + '\n', encoding='utf-8')
        print('Severe class-5 concentration:', severe, flush=True)
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14})
        for pair in range(2):
            fig, axes = plt.subplots(2, 1, figsize=(7, 10))
            for ax, method, name in zip(axes, METHODS[2 * pair:2 * pair + 2], NAMES[2 * pair:2 * pair + 2]):
                row = next(r for r in summary if r['method'] == method and r['condition'] == 'severe'
                           and r['train_seed'] == '11' and r['noise_seed'] == '101')
                matrix = matrices[row['run_id']]
                normalized = matrix / matrix.sum(1, keepdims=True)
                im = ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
                ax.set(title=name, xlabel='Lớp dự đoán', ylabel='Lớp thật', xticks=range(0, 43, 6), yticks=range(0, 43, 6))
                ax.tick_params(labelsize=13)
                fig.colorbar(im, ax=ax, fraction=.045, pad=.035).ax.tick_params(labelsize=12)
            fig.tight_layout(h_pad=1.6)
            fig.savefig(images / f'confusions_detail_{pair + 1}.png', dpi=200)
            plt.close(fig)
        with (evidence / 'per_class_aggregate.csv').open(encoding='utf-8', newline='') as stream:
            per_class = list(csv.DictReader(stream))
        fig, axes = plt.subplots(2, 1, figsize=(10, 7.2), sharey=True)
        for ax, lo, hi in zip(axes, [0, 22], [22, 43]):
            for method, name in zip(METHODS, NAMES):
                part = sorted([r for r in per_class if r['condition'] == 'severe' and r['method'] == method
                               and lo <= int(r['class_id']) < hi], key=lambda r: int(r['class_id']))
                ax.plot([int(r['class_id']) for r in part], [100 * float(r['f1_mean']) for r in part],
                        'o-', linewidth=1.5, markersize=4, label=name)
            ax.set(xlabel='Class ID', ylabel='F1 (%)', xticks=range(lo, hi), ylim=(0, 103))
            ax.tick_params(labelsize=12); ax.grid(alpha=.2)
        axes[0].legend(ncol=2, frameon=False, fontsize=12, loc='lower left')
        fig.tight_layout(h_pad=1.5)
        fig.savefig(images / 'per_class_severe_detail.png', dpi=200); plt.close(fig)
    outputs = [evidence / 'order_bootstrap.csv', evidence / 'class5_concentration.csv',
               tables / 'order_bootstrap.tex', tables / 'class5_concentration.tex'] + [
                   images / name for name in ['confusions_detail_1.png', 'confusions_detail_2.png', 'per_class_severe_detail.png']]
    metadata = dict(result_archive_sha256=audit['archive_sha256'], status='completed', script_sha256=digest(__file__),
                    prediction_rows_paired=int(predictions.size), images=len(labels), checkpoint_noise_pairs=run_keys,
                    bootstrap=dict(repetitions=args.repetitions, seed=args.seed, method='paired class-stratified image percentile bootstrap and paired hierarchical checkpoint/image bootstrap',
                                   resampling_unit='official test source image; same multiplicity for both methods and every fixed checkpoint/noise pair',
                                   aggregation='Macro-F1 per run, mean noise within checkpoint, then mean checkpoints',
                                   interval='marginal 95% percentile; three conditions, no multiplicity correction',
                                   checkpoint_resampling='Resample three training-seed clusters with replacement; preserve the three noise runs within each chosen checkpoint; same image multiplicities across all clusters and both methods',
                                   limitations='Only three training clusters, three fixed noises; image independence assumed, no verified official test tracks; not a population CI for real cameras'),
                    class_5_support=int(support[5]), class_5_true_percent=float(100 * support[5] / len(labels)),
                    labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),
                    paired_predictions_sha256=hashlib.sha256(predictions.tobytes()).hexdigest(),
                    files={p.relative_to(ROOT).as_posix(): digest(p) for p in outputs})
    (evidence / 'review_analysis_audit.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
