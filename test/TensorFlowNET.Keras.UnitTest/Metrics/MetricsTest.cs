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
