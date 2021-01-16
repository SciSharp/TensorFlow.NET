using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow.Keras.Engine;
using static Tensorflow.KerasApi;

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
            var new_model = keras.models.from_config(config);
            Assert.AreEqual(model.Layers.Count, new_model.Layers.Count);
        }

        Functional GetFunctionalModel()
        {
            // Create a simple model.
            var inputs = keras.Input(shape: 32);
            var dense_layer = keras.layers.Dense(1);
            var outputs = dense_layer.Apply(inputs);
            return keras.Model(inputs, outputs);
        }
    }
}
