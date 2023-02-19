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
}
