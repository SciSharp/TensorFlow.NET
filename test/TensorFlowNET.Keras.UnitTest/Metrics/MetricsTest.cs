using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.UnitTest.Layers.Metrics;

[TestClass]
public class MetricsTest : EagerModeTestBase
{
    /// <summary>
    /// https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Accuracy
    /// </summary>
    [TestMethod]
    public void Accuracy()
    {
        var y_true = np.array(new[,] { { 1 }, { 2 }, { 3 }, { 4 } });
        var y_pred = np.array(new[,] { { 0f }, { 2f }, { 3f }, { 4f } });
        var m = tf.keras.metrics.Accuracy();
        m.update_state(y_true, y_pred);
        var r = m.result().numpy();
        Assert.AreEqual(r, 0.75f);

        m.reset_states();
        var weights = np.array(new[] { 1f, 1f, 0f, 0f });
        m.update_state(y_true, y_pred, sample_weight: weights);
        r = m.result().numpy();
        Assert.AreEqual(r, 0.5f);
    }

    /// <summary>
    /// https://www.tensorflow.org/api_docs/python/tf/keras/metrics/BinaryAccuracy
    /// </summary>
    [TestMethod]
    public void BinaryAccuracy()
    {
        var y_true = np.array(new[,] { { 1 }, { 1 }, { 0 }, { 0 } });
        var y_pred = np.array(new[,] { { 0.98f }, { 1f }, { 0f }, { 0.6f } });
        var m = tf.keras.metrics.BinaryAccuracy();
        m.update_state(y_true, y_pred);
        var r = m.result().numpy();
        Assert.AreEqual(r, 0.75f);

        m.reset_states();
        var weights = np.array(new[] { 1f, 0f, 0f, 1f });
        m.update_state(y_true, y_pred, sample_weight: weights);
        r = m.result().numpy();
        Assert.AreEqual(r, 0.5f);
    }

    /// <summary>
    /// https://www.tensorflow.org/api_docs/python/tf/keras/metrics/CategoricalAccuracy
    /// </summary>
    [TestMethod]
    public void CategoricalAccuracy()
    {
        var y_true = np.array(new[,] { { 0, 0, 1 }, { 0, 1, 0 } });
        var y_pred = np.array(new[,] { { 0.1f, 0.9f, 0.8f }, { 0.05f, 0.95f, 0f } });
        var m = tf.keras.metrics.CategoricalAccuracy();
        m.update_state(y_true, y_pred);
        var r = m.result().numpy();
        Assert.AreEqual(r, 0.5f);

        m.reset_states();
        var weights = np.array(new[] { 0.7f, 0.3f });
        m.update_state(y_true, y_pred, sample_weight: weights);
        r = m.result().numpy();
        Assert.AreEqual(r, 0.3f);
    }

    /// <summary>
    /// https://www.tensorflow.org/api_docs/python/tf/keras/metrics/SparseCategoricalAccuracy
    /// </summary>
    [TestMethod]
    public void SparseCategoricalAccuracy()
    {
        var y_true = np.array(new[] { 2, 1 });
        var y_pred = np.array(new[,] { { 0.1f, 0.6f, 0.3f }, { 0.05f, 0.95f, 0f } });
        var m = tf.keras.metrics.SparseCategoricalAccuracy();
        m.update_state(y_true, y_pred);
        var r = m.result().numpy();
        Assert.AreEqual(r, 0.5f);

        m.reset_states();
        var weights = np.array(new[] { 0.7f, 0.3f });
        m.update_state(y_true, y_pred, sample_weight: weights);
        r = m.result().numpy();
        Assert.AreEqual(r, 0.3f);
    }

    /// <summary>
    /// https://www.tensorflow.org/api_docs/python/tf/keras/metrics/CategoricalCrossentropy
    /// </summary>
    [TestMethod]
    public void CategoricalCrossentropy()
    {
        var y_true = np.array(new[,] { { 0, 1, 0 }, { 0, 0, 1 } });
        var y_pred = np.array(new[,] { { 0.05f, 0.95f, 0f }, { 0.1f, 0.8f, 0.1f } });
        var m = tf.keras.metrics.CategoricalCrossentropy();
        m.update_state(y_true, y_pred);
        var r = m.result().numpy();
        Assert.AreEqual(r, 1.1769392f);

        m.reset_states();
        var weights = np.array(new[] { 0.3f, 0.7f });
        m.update_state(y_true, y_pred, sample_weight: weights);
        r = m.result().numpy();
        Assert.AreEqual(r, 1.6271976f);
    }

    /// <summary>
    /// https://www.tensorflow.org/api_docs/python/tf/keras/metrics/SparseCategoricalCrossentropy
    /// </summary>
    [TestMethod]
    public void SparseCategoricalCrossentropy()
    {
        var y_true = np.array(new[] { 1, 2 });
        var y_pred = np.array(new[,] { { 0.05f, 0.95f, 0f }, { 0.1f, 0.8f, 0.1f } });
        var m = tf.keras.metrics.SparseCategoricalCrossentropy();
        m.update_state(y_true, y_pred);
        var r = m.result().numpy();
        Assert.AreEqual(r, 1.1769392f);
    }

    /// <summary>
    /// https://www.tensorflow.org/api_docs/python/tf/keras/metrics/CosineSimilarity
    /// </summary>
    [TestMethod]
    public void CosineSimilarity()
    {
        var y_true = np.array(new[,] { { 0, 1 }, { 1, 1 } });
        var y_pred = np.array(new[,] { { 1f, 0f }, { 1f, 1f } });
        var m = tf.keras.metrics.CosineSimilarity(axis: 1);
        m.update_state(y_true, y_pred);
        var r = m.result().numpy();
        Assert.AreEqual(r, 0.49999997f);

        m.reset_states();
        var weights = np.array(new[] { 0.3f, 0.7f });
        m.update_state(y_true, y_pred, sample_weight: weights);
        r = m.result().numpy();
        Assert.AreEqual(r, 0.6999999f);
    }

    /// <summary>
    /// https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/F1Score
    /// </summary>
    [TestMethod]
    public void F1Score()
    {
        var y_true = np.array(new[,] { { 1, 1, 1 }, { 1, 0, 0 }, { 1, 1, 0 } });
        var y_pred = np.array(new[,] { { 0.2f, 0.6f, 0.7f }, { 0.2f, 0.6f, 0.6f }, { 0.6f, 0.8f, 0f } });
        var m = tf.keras.metrics.F1Score(num_classes: 3, threshold: 0.5f);
        m.update_state(y_true, y_pred);
        var r = m.result().numpy();
        Assert.AreEqual(r, new[] { 0.5f, 0.8f, 0.6666667f });
    }

    /// <summary>
    /// https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/FBetaScore
    /// </summary>
    [TestMethod]
    public void FBetaScore()
    {
        var y_true = np.array(new[,] { { 1, 1, 1 }, { 1, 0, 0 }, { 1, 1, 0 } });
        var y_pred = np.array(new[,] { { 0.2f, 0.6f, 0.7f }, { 0.2f, 0.6f, 0.6f }, { 0.6f, 0.8f, 0f } });
        var m = tf.keras.metrics.FBetaScore(num_classes: 3, beta: 2.0f, threshold: 0.5f);
        m.update_state(y_true, y_pred);
        var r = m.result().numpy();
        Assert.AreEqual(r, new[] { 0.3846154f, 0.90909094f, 0.8333334f });
    }

    /// <summary>
    /// https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/HammingLoss
    /// </summary>
    [TestMethod]
    public void HammingLoss()
    {
        // multi-class hamming loss
        var y_true = np.array(new[,]
        {
            { 1, 0, 0, 0 },
            { 0, 0, 1, 0 },
            { 0, 0, 0, 1 },
            { 0, 1, 0, 0 }
        });
        var y_pred = np.array(new[,]
        {
            { 0.8f, 0.1f, 0.1f, 0.0f },
            { 0.2f, 0.0f, 0.8f, 0.0f },
            { 0.05f, 0.05f, 0.1f, 0.8f },
            { 1.0f, 0.0f, 0.0f, 0.0f }
        });
        var m = tf.keras.metrics.HammingLoss(mode: "multiclass", threshold: 0.6f);
        m.update_state(y_true, y_pred);
        var r = m.result().numpy();
        Assert.AreEqual(r, 0.25f);

        // multi-label hamming loss
        y_true = np.array(new[,]
        {
            { 1, 0, 1, 0 },
            { 0, 1, 0, 1 },
            { 0, 0, 0, 1 }
        });
        y_pred = np.array(new[,]
        {
            { 0.82f, 0.5f, 0.9f, 0.0f },
            { 0f, 1f, 0.4f, 0.98f },
            { 0.89f, 0.79f, 0f, 0.3f }
        });
        m = tf.keras.metrics.HammingLoss(mode: "multilabel", threshold: 0.8f);
        m.update_state(y_true, y_pred);
        r = m.result().numpy();
        Assert.AreEqual(r, 0.16666667f);
    }

    /// <summary>
    /// https://www.tensorflow.org/api_docs/python/tf/keras/metrics/TopKCategoricalAccuracy
    /// </summary>
    [TestMethod]
    public void TopKCategoricalAccuracy()
    {
        var y_true = np.array(new[,] { { 0, 0, 1 }, { 0, 1, 0 } });
        var y_pred = np.array(new[,] { { 0.1f, 0.9f, 0.8f }, { 0.05f, 0.95f, 0f } });
        var m = tf.keras.metrics.TopKCategoricalAccuracy(k: 1);
        m.update_state(y_true, y_pred);
        var r = m.result().numpy();
        Assert.AreEqual(r, 0.5f);

        m.reset_states();
        var weights = np.array(new[] { 0.7f, 0.3f });
        m.update_state(y_true, y_pred, sample_weight: weights);
        r = m.result().numpy();
        Assert.AreEqual(r, 0.3f);
    }

    /// <summary>
    /// https://www.tensorflow.org/api_docs/python/tf/keras/metrics/SparseTopKCategoricalAccuracy
    /// </summary>
    [TestMethod]
    public void SparseTopKCategoricalAccuracy()
    {
        var y_true = np.array(new[] { 2, 1 });
        var y_pred = np.array(new[,] { { 0.1f, 0.9f, 0.8f }, { 0.05f, 0.95f, 0f } });
        var m = tf.keras.metrics.SparseTopKCategoricalAccuracy(k: 1);
        m.update_state(y_true, y_pred);
        var r = m.result().numpy();
        Assert.AreEqual(r, 0.5f);

        m.reset_states();
        var weights = np.array(new[] { 0.7f, 0.3f });
        m.update_state(y_true, y_pred, sample_weight: weights);
        r = m.result().numpy();
        Assert.AreEqual(r, 0.3f);
    }

    /// <summary>
    /// https://www.tensorflow.org/api_docs/python/tf/keras/metrics/top_k_categorical_accuracy
    /// </summary>
    [TestMethod]
    public void top_k_categorical_accuracy()
    {
        var y_true = np.array(new[,] { { 0, 0, 1 }, { 0, 1, 0 } });
        var y_pred = np.array(new[,] { { 0.1f, 0.9f, 0.8f }, { 0.05f, 0.95f, 0f } });
        var m = tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k: 3);
        Assert.AreEqual(m.numpy(), new[] { 1f, 1f });
    }

    /// <summary>
    /// https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Precision
    /// </summary>
    [TestMethod]
    public void Precision()
    {
        var y_true = np.array(new[] { 0, 1, 1, 1 });
        var y_pred = np.array(new[] { 1, 0, 1, 1 });
        var m = tf.keras.metrics.Precision();
        m.update_state(y_true, y_pred);
        var r = m.result().numpy();
        Assert.AreEqual(r, 0.6666667f);

        m.reset_states();
        var weights = np.array(new[] { 0f, 0f, 1f, 0f });
        m.update_state(y_true, y_pred, sample_weight: weights);
        r = m.result().numpy();
        Assert.AreEqual(r, 1f);

        // With top_k=2, it will calculate precision over y_true[:2]
        // and y_pred[:2]
        m = tf.keras.metrics.Precision(top_k: 2);
        m.update_state(np.array(new[] { 0, 0, 1, 1 }), np.array(new[] { 1, 1, 1, 1 }));
        r = m.result().numpy();
        Assert.AreEqual(r, 0f);

        // With top_k=4, it will calculate precision over y_true[:4]
        // and y_pred[:4]
        m = tf.keras.metrics.Precision(top_k: 4);
        m.update_state(np.array(new[] { 0, 0, 1, 1 }), np.array(new[] { 1, 1, 1, 1 }));
        r = m.result().numpy();
        Assert.AreEqual(r, 0.5f);
    }

    /// <summary>
    /// https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Recall
    /// </summary>
    [TestMethod]
    public void Recall()
    {
        var y_true = np.array(new[] { 0, 1, 1, 1 });
        var y_pred = np.array(new[] { 1, 0, 1, 1 });
        var m = tf.keras.metrics.Recall();
        m.update_state(y_true, y_pred);
        var r = m.result().numpy();
        Assert.AreEqual(r, 0.6666667f);

        m.reset_states();
        var weights = np.array(new[] { 0f, 0f, 1f, 0f });
        m.update_state(y_true, y_pred, sample_weight: weights);
        r = m.result().numpy();
        Assert.AreEqual(r, 1f);
    }
}
