using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlowNET.Keras.UnitTest;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.UnitTest.Losses;

[TestClass]
public class LossesTest : EagerModeTestBase
{
    /// <summary>
    /// https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy
    /// </summary>
    [TestMethod]
    public void BinaryCrossentropy()
    {
        // Example 1: (batch_size = 1, number of samples = 4)
        var y_true = tf.constant(new float[] { 0, 1, 0, 0 });
        var y_pred = tf.constant(new float[] { -18.6f, 0.51f, 2.94f, -12.8f });
        var bce = tf.keras.losses.BinaryCrossentropy(from_logits: true);
        var loss = bce.Call(y_true, y_pred);
        Assert.AreEqual((float)loss, 0.865458f);

        // Example 2: (batch_size = 2, number of samples = 4)
        y_true = tf.constant(new float[,] { { 0, 1 }, { 0, 0 } });
        y_pred = tf.constant(new float[,] { { -18.6f, 0.51f }, { 2.94f, -12.8f } });
        bce = tf.keras.losses.BinaryCrossentropy(from_logits: true);
        loss = bce.Call(y_true, y_pred);
        Assert.AreEqual((float)loss, 0.865458f);

        // Using 'sample_weight' attribute
        loss = bce.Call(y_true, y_pred, sample_weight: tf.constant(new[] { 0.8f, 0.2f }));
        Assert.AreEqual((float)loss, 0.2436386f);

        // Using 'sum' reduction` type.
        bce = tf.keras.losses.BinaryCrossentropy(from_logits: true, reduction: Reduction.SUM);
        loss = bce.Call(y_true, y_pred);
        Assert.AreEqual((float)loss, 1.730916f);

        // Using 'none' reduction type.
        bce = tf.keras.losses.BinaryCrossentropy(from_logits: true, reduction: Reduction.NONE);
        loss = bce.Call(y_true, y_pred);
        Assert.AreEqual(new float[] { 0.23515666f, 1.4957594f}, loss.numpy());
    }
}
