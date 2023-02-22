using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Keras.UnitTest;

[TestClass]
public class MetricsTest : EagerModeTestBase
{
    /// <summary>
    /// https://www.tensorflow.org/api_docs/python/tf/keras/metrics/BinaryAccuracy
    /// </summary>
    [TestMethod]
    public void BinaryAccuracy()
    {
        var y_true = np.array(new[,] { { 1 }, { 1 },{ 0 }, { 0 } });
        var y_pred = np.array(new[,] { { 0.98f }, { 1f }, { 0f }, { 0.6f } });
        var m = tf.keras.metrics.BinaryAccuracy();
        /*m.update_state(y_true, y_pred);
        var r = m.result().numpy();
        Assert.AreEqual(r, 0.75f);

        m.reset_states();*/
        var weights = np.array(new[] { 1f, 0f, 0f, 1f });
        m.update_state(y_true, y_pred, sample_weight: weights);
        var r = m.result().numpy();
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
