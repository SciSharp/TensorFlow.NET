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
        public void ReLU()
        {
            var layer = keras.layers.ReLU();
            Tensor output = layer.Apply(np.array(-3.0f, -1.0f, 0.0f, 2.0f));
            Equal(new[] { 0.0f, 0.0f, 0.0f, 2.0f }, output.ToArray<float>());
        }

        [TestMethod]
        public void Sigmoid()
        {
            var layer = keras.layers.Sigmoid();
            Tensor output = layer.Apply(np.array(-3.0f, -1.0f, 0.0f, 2.0f));
            Equal(new[] { 0.047425866f, 0.26894143f, 0.5f, 0.8807971f }, output.ToArray<float>());
        }
        
        [TestMethod]
        public void Tanh()
        {
            var layer = keras.layers.Tanh();
            Tensor output = layer.Apply(np.array(-3.0f, -1.0f, 0.0f, 2.0f));
            // {-0.9950547f, -0.7615942f, 0f, 0.9640276f}
            Equal(new[] { -0.9950547f, -0.7615942f, 0f, 0.9640276f }, output.ToArray<float>());
        }

        [TestMethod]
        public void LeakyReLU()
        {
            var layer = keras.layers.LeakyReLU();
            Tensor output = layer.Apply(np.array(-3.0f, -1.0f, 0.0f, 2.0f));
            Equal(new[] { -0.9f, -0.3f, 0.0f, 2.0f }, output.ToArray<float>());
        }
    }
}
