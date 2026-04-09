# coding=utf-8
# Copyright 2026 The init2winit Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for metrics.py.

"""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from init2winit.model_lib import losses
from init2winit.model_lib import metrics
import jax.numpy as jnp
import numpy as np
from skimage.metrics import structural_similarity as ssim


class MetricsTest(parameterized.TestCase):
  """Tests for metrics.py."""

  @parameterized.named_parameters(
      dict(
          testcase_name='basic',
          targets=np.array([[1., 0.], [0., 1.]]),
          logits=np.array([[0.5, 0.5], [0.5, 0.5]]),
          weights=np.array([[1., 1.], [1., 1.]]),
          result=0.5),
      dict(
          testcase_name='weights',
          targets=np.array([[1., 0.,], [0., 1.], [0., 1.]]),
          logits=np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.7]]),
          weights=np.array([[1., 1.], [0., 1.], [1., 0.]]),
          result=0.5))
  def test_map(self, logits, targets, weights, result):
    """Tests the mean average precision computation."""

    average_precision = metrics.OGBGMeanAveragePrecision.from_model_output(
        logits=logits, targets=targets, weights=weights).compute()
    self.assertAlmostEqual(average_precision, result)

  def test_structural_similarity(self):
    """Tests jax implementation of structural similarity.

    NOTE(dsuo): we test with the defaults used in the FastMRI workload.
    """
    im1 = np.array(
        [[
            0.73833251, 0.89810601, 0.59434839, 0.112503, 0.40403852,
            0.05790091, 0.81124133, 0.68376994, 0.58584383, 0.0930026
        ],
         [
             0.42542345, 0.56915387, 0.17478424, 0.1589461, 0.74287562,
             0.47219216, 0.52649117, 0.50070807, 0.67500359, 0.37819205
         ],
         [
             0.68373459, 0.64230722, 0.97335725, 0.24565012, 0.48942928,
             0.50963254, 0.37571989, 0.38366919, 0.36945232, 0.09163938
         ],
         [
             0.15319016, 0.36161473, 0.61484123, 0.17523618, 0.73859486,
             0.16115077, 0.01884255, 0.98497526, 0.02232614, 0.71922009
         ],
         [
             0.5019574, 0.80491521, 0.65586547, 0.03463707, 0.47130842,
             0.63220364, 0.76905247, 0.50815002, 0.92499088, 0.20647629
         ],
         [
             0.77917087, 0.07486334, 0.36286554, 0.27815142, 0.706411,
             0.41677936, 0.23959606, 0.51440788, 0.75697984, 0.80130235
         ],
         [
             0.55735541, 0.62024555, 0.87929081, 0.8054033, 0.12014468,
             0.22865017, 0.23542662, 0.71521724, 0.40843243, 0.25604842
         ],
         [
             0.26829756, 0.19476007, 0.38425566, 0.38231672, 0.12902957,
             0.60572083, 0.65571312, 0.6134444, 0.13835472, 0.06307113
         ],
         [
             0.75307163, 0.44877311, 0.31321154, 0.03057511, 0.86964725,
             0.89864268, 0.53593918, 0.87913734, 0.72179573, 0.03976076
         ],
         [
             0.55826683, 0.18015783, 0.90069321, 0.55805617, 0.71336459,
             0.67812158, 0.27842012, 0.85890605, 0.22515742, 0.60356573
         ]])

    im2 = np.array(
        [[
            0.38785895, 0.69382307, 0.24389369, 0.89767306, 0.42301789,
            0.16313277, 0.80090617, 0.16567136, 0.71543147, 0.30399568
        ],
         [
             0.35219695, 0.57636231, 0.5339162, 0.51421423, 0.55444482,
             0.3299572, 0.08871051, 0.90975499, 0.26302511, 0.08448494
         ],
         [
             0.88601557, 0.41470639, 0.68370194, 0.64813528, 0.86429226,
             0.69276718, 0.66361842, 0.4851298, 0.74617258, 0.28851107
         ],
         [
             0.84669316, 0.99759206, 0.79429959, 0.19977481, 0.4833177,
             0.57696104, 0.13978823, 0.63513837, 0.73423608, 0.83064902
         ],
         [
             0.45382891, 0.41018542, 0.86997271, 0.39990761, 0.32097822,
             0.52282046, 0.05960004, 0.95429451, 0.03181412, 0.80956527
         ],
         [
             0.90649511, 0.61557879, 0.59897015, 0.94188484, 0.90297625,
             0.76986281, 0.2392755, 0.33402192, 0.36923513, 0.54177217
         ],
         [
             0.44340055, 0.58440755, 0.45363187, 0.74527457, 0.23761691,
             0.74693863, 0.11449182, 0.48795747, 0.94897711, 0.01631275
         ],
         [
             0.30310764, 0.07203944, 0.11931363, 0.48873794, 0.18900569,
             0.00643777, 0.30659393, 0.41300417, 0.69529398, 0.24826242
         ],
         [
             0.98076131, 0.6875125, 0.54545994, 0.16997529, 0.0698003,
             0.59835326, 0.58198102, 0.8785474, 0.69644425, 0.73404286
         ],
         [
             0.1310947, 0.91694649, 0.32005394, 0.98112882, 0.4818337,
             0.26479291, 0.97803938, 0.03502056, 0.72615619, 0.72047081
         ]])

    jim1 = jnp.array(im1)
    jim2 = jnp.array(im2)

    expected = ssim(im1, im2, data_range=1.0)
    result = metrics.structural_similarity(jim1, jim2, data_range=1.0).item()

    self.assertAlmostEqual(expected, result)

  def test_wer_spm(self):
    """Tests word error rate metric implementation with SPM tokenizer."""

    source_sentence = "Let's start      praying this test passes!"
    decoded_sentence = source_sentence

    class MockSPMTokenizer():

      def tokenize(self, s):
        return s

      def detokenize(self, s):
        return s

    tokenizer = MockSPMTokenizer()
    tokenizer_type = 'SPM'

    with mock.patch.object(
        tokenizer, 'tokenize', return_value=[1, 2, 3], autospec=True):
      with mock.patch.object(
          tokenizer,
          'detokenize',
          return_value=[source_sentence],
          autospec=True):
        source_tokens = jnp.array([tokenizer.tokenize(source_sentence)],
                                  dtype=jnp.int32)
        source_paddings = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)

        decoded_tokens = jnp.array([tokenizer.tokenize(decoded_sentence)],
                                   dtype=jnp.int32)
        decoded_paddings = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)

        word_errors, num_words = metrics.compute_wer(decoded_tokens,
                                                     decoded_paddings,
                                                     source_tokens,
                                                     source_paddings, tokenizer,
                                                     tokenizer_type)
        self.assertEqual(word_errors, 0)
        self.assertEqual(num_words, 6.0)


class NumExamplesTest(absltest.TestCase):
  """Tests for the NumExamples metric."""

  def test_counts_without_weights(self):
    """Without weights, counts the batch size."""
    m = metrics.NumExamples.from_model_output(
        logits=jnp.ones((8, 10)), targets=jnp.ones((8, 10)), weights=None
    )
    np.testing.assert_allclose(m.compute(), 8.0)

  def test_counts_weighted_examples(self):
    """With weights, computes the sum of weights."""
    m = metrics.NumExamples.from_model_output(
        logits=jnp.ones((5, 3)),
        targets=jnp.ones((5, 3)),
        weights=jnp.array([1.0, 1.0, 0.0, 0.0, 1.0]),
    )
    np.testing.assert_allclose(m.compute(), 3.0)

  def test_merge_sums_counts(self):
    """Merging two metrics sums their counts."""
    m1 = metrics.NumExamples.from_model_output(
        logits=jnp.ones((3, 2)), targets=jnp.ones((3, 2)), weights=jnp.ones(3)
    )
    m2 = metrics.NumExamples.from_model_output(
        logits=jnp.ones((4, 2)), targets=jnp.ones((4, 2)), weights=jnp.ones(4)
    )
    np.testing.assert_allclose(m1.merge(m2).compute(), 7.0)


class CollectingMetricTest(absltest.TestCase):
  """Tests for CollectingMetric and from_outputs."""

  def test_stores_and_computes(self):
    """from_model_output stores values; compute() concatenates them."""
    cls = metrics.CollectingMetric.from_outputs(('x',))
    m1 = cls.from_model_output(x=jnp.array([1.0, 2.0]))
    m2 = cls.from_model_output(x=jnp.array([3.0, 4.0]))
    result = m1.merge(m2).compute()
    np.testing.assert_array_equal(result['x'], [1.0, 2.0, 3.0, 4.0])

  def test_multidimensional_concat(self):
    """2D arrays are concatenated along axis 0."""
    cls = metrics.CollectingMetric.from_outputs(('x',))
    m1 = cls.from_model_output(x=jnp.ones((2, 3)))
    m2 = cls.from_model_output(x=jnp.ones((4, 3)))
    self.assertEqual(m1.merge(m2).compute()['x'].shape, (6, 3))


class CollectionTest(absltest.TestCase):
  """Tests for Collection.create and the collection lifecycle."""

  def _make_cls(self):
    return metrics.Collection.create(
        error_rate=metrics.weighted_average_metric(
            metrics.weighted_misclassifications
        ),
        ce_loss=metrics.weighted_average_metric(
            losses.weighted_unnormalized_cross_entropy
        ),
        num_examples=metrics.NumExamples,
    )

  def test_compute_returns_all_metric_keys(self):
    """compute() returns a dict with one entry per metric."""
    cls = self._make_cls()
    c = cls.single_from_model_output(
        logits=jnp.array([[0.0, 1.0]]),
        targets=jnp.array([[0.0, 1.0]]),
        weights=jnp.array([1.0]),
    )
    self.assertEqual(
        set(c.compute().keys()), {'error_rate', 'ce_loss', 'num_examples'}
    )

  def test_perfect_and_full_misclassification(self):
    """Perfect predictions yield 0% error; wrong yield 100%."""
    cls = self._make_cls()
    perfect = cls.single_from_model_output(
        logits=jnp.array([[0.0, 100.0], [100.0, 0.0]]),
        targets=jnp.array([[0.0, 1.0], [1.0, 0.0]]),
        weights=jnp.array([1.0, 1.0]),
    )
    wrong = cls.single_from_model_output(
        logits=jnp.array([[100.0, 0.0], [0.0, 100.0]]),
        targets=jnp.array([[0.0, 1.0], [1.0, 0.0]]),
        weights=jnp.array([1.0, 1.0]),
    )
    np.testing.assert_allclose(perfect.compute()['error_rate'], 0.0)
    np.testing.assert_allclose(wrong.compute()['error_rate'], 1.0)

  def test_merge_accumulates(self):
    """Merging two collections produces correct averages."""
    cls = self._make_cls()
    c1 = cls.single_from_model_output(
        logits=jnp.array([[0.0, 100.0], [100.0, 0.0]]),
        targets=jnp.array([[0.0, 1.0], [1.0, 0.0]]),
        weights=jnp.array([1.0, 1.0]),
    )
    c2 = cls.single_from_model_output(
        logits=jnp.array([[100.0, 0.0]]),
        targets=jnp.array([[0.0, 1.0]]),
        weights=jnp.array([1.0]),
    )
    result = c1.merge(c2).compute()
    np.testing.assert_allclose(result['num_examples'], 3.0)
    np.testing.assert_allclose(result['error_rate'], 1.0 / 3.0, rtol=1e-5)

  def test_different_batch_sizes(self):
    """Aggregation is correct across batches of sizes 1, 4, and 8."""
    cls = self._make_cls()
    b1 = cls.single_from_model_output(
        logits=jnp.array([[0.0, 100.0]]),
        targets=jnp.array([[0.0, 1.0]]),
        weights=jnp.array([1.0]),
    )
    b4 = cls.single_from_model_output(
        logits=jnp.tile(jnp.array([[100.0, 0.0]]), (4, 1)),
        targets=jnp.tile(jnp.array([[0.0, 1.0]]), (4, 1)),
        weights=jnp.ones(4),
    )
    b8 = cls.single_from_model_output(
        logits=jnp.tile(jnp.array([[0.0, 100.0]]), (8, 1)),
        targets=jnp.tile(jnp.array([[0.0, 1.0]]), (8, 1)),
        weights=jnp.ones(8),
    )
    result = b1.merge(b4).merge(b8).compute()
    np.testing.assert_allclose(result['num_examples'], 13.0)
    np.testing.assert_allclose(result['error_rate'], 4.0 / 13.0, rtol=1e-5)

  def test_multi_batch_eval_loop(self):
    """Simulates a multiple batch size eval loop."""
    cls = self._make_cls()
    batches = [
        dict(
            logits=jnp.array([[0.0, 1.0], [1.0, 0.0]]),
            targets=jnp.array([[0.0, 1.0], [1.0, 0.0]]),
            weights=jnp.array([1.0, 1.0]),
        ),
        dict(
            logits=jnp.array([[1.0, 0.0]]),
            targets=jnp.array([[0.0, 1.0]]),
            weights=jnp.array([1.0]),
        ),
        dict(
            logits=jnp.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
            targets=jnp.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
            weights=jnp.array([1.0, 1.0, 1.0]),
        ),
    ]
    collection = None
    for batch in batches:
      update = cls.single_from_model_output(**batch)
      collection = update if collection is None else collection.merge(update)
    result = collection.compute()
    np.testing.assert_allclose(result['num_examples'], 6.0)
    np.testing.assert_allclose(result['error_rate'], 1.0 / 6.0, rtol=1e-5)

  def test_with_collecting_metric_end_to_end(self):
    """Collection containing a CollectingMetric computes all keys."""
    cls = metrics.Collection.create(
        ce_loss=metrics.weighted_average_metric(
            losses.unnormalized_sigmoid_binary_cross_entropy
        ),
        num_examples=metrics.NumExamples,
        average_precision=metrics.OGBGMeanAveragePrecision,
    )
    c = cls.single_from_model_output(
        logits=jnp.array([[0.5, 0.5], [0.5, 0.5]]),
        targets=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
        weights=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
    )
    result = c.compute()
    self.assertEqual(
        set(result.keys()), {'ce_loss', 'num_examples', 'average_precision'}
    )


class OGBGMeanAveragePrecisionUnitTest(absltest.TestCase):
  """Unit tests for OGBGMeanAveragePrecision."""

  def test_perfect_predictions(self):
    """Perfect predictions yield mAP of 1.0."""
    m = metrics.OGBGMeanAveragePrecision.from_model_output(
        logits=jnp.array([[100.0, -100.0], [-100.0, 100.0]]),
        targets=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
        weights=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
    )
    np.testing.assert_allclose(m.compute(), 1.0, rtol=1e-5)

  def test_all_same_class_returns_nan(self):
    """If all targets are the same class, mAP is NaN."""
    m = metrics.OGBGMeanAveragePrecision.from_model_output(
        logits=jnp.array([[0.5, 0.5], [0.3, 0.3]]),
        targets=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
        weights=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
    )
    self.assertTrue(np.isnan(m.compute()))

  def test_masked_weights_change_result(self):
    """Masking out a wrong prediction improves mAP."""
    logits = jnp.array([[100.0, -100.0], [-100.0, 100.0], [-100.0, 100.0]])
    targets = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    m_full = metrics.OGBGMeanAveragePrecision.from_model_output(
        logits=logits,
        targets=targets,
        weights=jnp.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
    )
    m_partial = metrics.OGBGMeanAveragePrecision.from_model_output(
        logits=logits,
        targets=targets,
        weights=jnp.array([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]),
    )
    self.assertLess(m_full.compute(), m_partial.compute())


class WeightedAverageMetricUnitTest(absltest.TestCase):
  """Tests for the weighted_average_metric factory."""

  def test_correct_average(self):
    """Computes correct weighted average for a single batch."""
    cls = metrics.weighted_average_metric(metrics.weighted_misclassifications)
    m = cls.from_model_output(
        logits=jnp.array([[0.0, 1.0], [1.0, 0.0]]),
        targets=jnp.array([[0.0, 1.0], [1.0, 0.0]]),
        weights=jnp.array([1.0, 1.0]),
    )
    np.testing.assert_allclose(m.compute(), 0.0)

  def test_merge_accumulates(self):
    """Merge sums total and weight; compute returns global average."""
    cls = metrics.weighted_average_metric(metrics.weighted_misclassifications)
    m1 = cls.from_model_output(
        logits=jnp.array([[0.0, 1.0], [1.0, 0.0]]),
        targets=jnp.array([[0.0, 1.0], [1.0, 0.0]]),
        weights=jnp.array([1.0, 1.0]),
    )
    m2 = cls.from_model_output(
        logits=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
        targets=jnp.array([[0.0, 1.0], [1.0, 0.0]]),
        weights=jnp.array([1.0, 1.0]),
    )
    np.testing.assert_allclose(m1.merge(m2).compute(), 0.5)


class PerplexityUnitTest(absltest.TestCase):
  """Tests for perplexity_fun."""

  def test_perplexity_is_exp_of_cross_entropy(self):
    """Perplexity equals exp(cross-entropy)."""
    cls = metrics.perplexity_fun()
    logits = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    targets = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    weights = jnp.array([1.0, 1.0])
    m = cls.from_model_output(logits=logits, targets=targets, weights=weights)
    ce = (
        losses.weighted_unnormalized_cross_entropy(
            logits, targets, weights
        ).sum()
        / weights.sum()
    )
    np.testing.assert_allclose(m.compute(), jnp.exp(ce), rtol=1e-5)


class MDLMMetricsTest(absltest.TestCase):
  """Tests for the MDLM metrics bundle."""

  def test_mdlm_bundle(self):
    """MDLM bundle passes through correct metrics."""
    bundle = metrics.get_metrics('mdlm_metrics')
    result = bundle.single_from_model_output(
        normalized_loss=jnp.float32(2.0),
        num_tokens=jnp.float32(100.0),
        effective_num_tokens=jnp.float32(50.0),
    ).compute()
    np.testing.assert_allclose(result['ce_loss'], 2.0)
    np.testing.assert_allclose(result['perplexity'], jnp.exp(2.0), rtol=1e-5)
    np.testing.assert_allclose(result['num_tokens'], 100.0)
    np.testing.assert_allclose(result['effective_num_tokens'], 50.0)


class MetricsBundleRegistryTest(parameterized.TestCase):
  """Tests for the metric bundle registry (get_metrics)."""

  @parameterized.parameters(
      'classification_metrics',
      'autoregressive_language_model_metrics',
  )
  def test_bundle_merge_and_compute(self, metrics_name):
    """Bundles can be merged, computed, and produce correct num_examples."""
    bundle = metrics.get_metrics(metrics_name)
    c1 = bundle.single_from_model_output(
        logits=jnp.array([[0.0, 1.0], [1.0, 0.0]]),
        targets=jnp.array([[0.0, 1.0], [1.0, 0.0]]),
        weights=jnp.array([1.0, 1.0]),
    )
    c2 = bundle.single_from_model_output(
        logits=jnp.array([[0.5, 0.5]]),
        targets=jnp.array([[0.0, 1.0]]),
        weights=jnp.array([1.0]),
    )
    result = c1.merge(c2).compute()
    self.assertIn('num_examples', result)
    np.testing.assert_allclose(result['num_examples'], 3.0)


if __name__ == '__main__':
  absltest.main()
