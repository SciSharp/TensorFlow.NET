using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow.Keras.Engine;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.UnitTest.Keras
{
    /// <summary>
    /// https://www.tensorflow.org/guide/keras/save_and_serialize
    /// </summary>
    [TestClass]
    public class ModelSaveTest : EagerModeTestBase
    {
        [TestMethod, Ignore]
        public void GetAndFromConfig()
        {
            var model = GetFunctionalModel();
            var config = model.get_config();
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
