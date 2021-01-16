using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;
using static Tensorflow.KerasApi;
using Tensorflow;

namespace TensorFlowNET.Keras.UnitTest
{
    [TestClass]
    public class ActivationTest : EagerModeTestBase
    {
        [TestMethod]
        public void LeakyReLU()
        {
            var layer = keras.layers.LeakyReLU();
            Tensor output = layer.Apply(np.array(-3.0f, -1.0f, 0.0f, 2.0f));
            Equal(new[] { -0.9f, -0.3f, 0.0f, 2.0f }, output.ToArray<float>());
        }
    }
}
