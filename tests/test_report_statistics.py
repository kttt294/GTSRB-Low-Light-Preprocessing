"""Statistical checks for paired report analysis, independent of model inference."""
import unittest
import numpy as np
from sklearn.metrics import f1_score

from report.latex.review_analysis import macro_scores, paired_bootstrap


class ReportBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.repeat(np.arange(43), 3)
        self.before = np.stack([self.labels, np.roll(self.labels, 1)])
        self.after = self.before.copy()
        self.after[1, ::7] = self.labels[::7]

    def test_weighted_scores_equal_literal_resampling(self):
        weights = np.arange(len(self.labels)) % 4
        indices = np.repeat(np.arange(len(weights)), weights)
        expected = [f1_score(self.labels[indices], r[indices], labels=range(43), average='macro', zero_division=0)
                    for r in self.after]
        np.testing.assert_allclose(macro_scores(self.labels, self.after, weights), expected, atol=1e-14)

    def test_identical_methods_have_exactly_zero_interval(self):
        point, draws = paired_bootstrap(self.labels, self.before, self.before, 30, 7)
        self.assertEqual(point, 0)
        np.testing.assert_array_equal(draws, 0)

    def test_duplicated_fixed_runs_do_not_inflate_image_sample_size(self):
        p, d = paired_bootstrap(self.labels, self.before, self.after, 40, 9)
        pp, dd = paired_bootstrap(self.labels, np.tile(self.before, (3, 1)), np.tile(self.after, (3, 1)), 40, 9)
        self.assertAlmostEqual(p, pp, places=12)
        np.testing.assert_allclose(d, dd, atol=1e-12)

    def test_swapping_methods_reverses_each_paired_difference(self):
        p, d = paired_bootstrap(self.labels, self.before, self.after, 30, 11)
        q, e = paired_bootstrap(self.labels, self.after, self.before, 30, 11)
        self.assertEqual(p, -q)
        np.testing.assert_array_equal(d, -e)

    def test_perfect_against_always_wrong_has_fixed_difference(self):
        good = self.labels[None, :]
        bad = ((self.labels + 1) % 43)[None, :]
        p, d = paired_bootstrap(self.labels, good, bad, 20, 1)
        self.assertEqual(p, -100)
        np.testing.assert_array_equal(d, -100)

    def test_single_checkpoint_cluster_preserves_image_only_interval(self):
        p, d = paired_bootstrap(self.labels, self.before, self.after, 30, 13)
        q, e = paired_bootstrap(self.labels, self.before, self.after, 30, 13, checkpoint_groups=[11, 11])
        self.assertEqual(p, q)
        np.testing.assert_array_equal(d, e)

    def test_checkpoint_clusters_keep_duplicates_paired(self):
        p, d = paired_bootstrap(self.labels, self.before, self.after, 30, 13, checkpoint_groups=[11, 22])
        q, e = paired_bootstrap(self.labels, np.tile(self.before, (3, 1)), np.tile(self.after, (3, 1)), 30, 13,
                                checkpoint_groups=[11, 22, 11, 22, 11, 22])
        self.assertAlmostEqual(p, q, places=12)
        np.testing.assert_allclose(d, e, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
