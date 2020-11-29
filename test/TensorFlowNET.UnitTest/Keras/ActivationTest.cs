using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;
using static Tensorflow.KerasApi;
using Tensorflow;

namespace TensorFlowNET.UnitTest.Keras
{
    [TestClass]
    public class ActivationTest : EagerModeTestBase
    {
        [TestMethod]
        public void LeakyReLU()
        {
            var layer = keras.layers.LeakyReLU();
            Tensor output = layer.Apply(np.array(-3.0f, -1.0f, 0.0f, 2.0f));
            var outputArray = output.ToArray<float>();
            assertFloat32Equal(-0.9f, outputArray[0], "Not equal");
            assertFloat32Equal(-0.3f, outputArray[1], "Not equal");
            assertFloat32Equal(0.0f, outputArray[2], "Not equal");
            assertFloat32Equal(2.0f, outputArray[3], "Not equal");
        }
    }
}
