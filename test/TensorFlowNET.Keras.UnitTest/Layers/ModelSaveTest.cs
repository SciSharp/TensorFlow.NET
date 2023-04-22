using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow.Keras.Engine;
using System.Diagnostics;
using static Tensorflow.KerasApi;
using Tensorflow.Keras.Saving;
using Tensorflow.Keras.Models;

namespace TensorFlowNET.Keras.UnitTest
{
    /// <summary>
    /// https://www.tensorflow.org/guide/keras/save_and_serialize
    /// </summary>
    [TestClass]
    public class ModelSaveTest : EagerModeTestBase
    {
        [TestMethod]
        public void GetAndFromConfig()
        {
            var model = GetFunctionalModel();
            var config = model.get_config();
            Debug.Assert(config is FunctionalConfig);
            var new_model = new ModelsApi().from_config(config as FunctionalConfig);
            Assert.AreEqual(model.Layers.Count, new_model.Layers.Count);
        }

        IModel GetFunctionalModel()
        {
            // Create a simple model.
            var inputs = keras.Input(shape: 32);
            var dense_layer = keras.layers.Dense(1);
            var outputs = dense_layer.Apply(inputs);
            return keras.Model(inputs, outputs);
        }
    }
}
