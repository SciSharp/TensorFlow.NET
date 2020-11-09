using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow.Keras.Engine;
using static Tensorflow.KerasExt;

namespace TensorFlowNET.UnitTest.Keras
{
    /// <summary>
    /// https://www.tensorflow.org/guide/keras/save_and_serialize
    /// </summary>
    [TestClass]
    public class ModelSaveTest : EagerModeTestBase
    {
        [TestMethod]
        public void SaveAndLoadTest()
        {
            var model = GetModel();
        }

        Model GetModel()
        {
            // Create a simple model.
            var inputs = keras.Input(shape: 32);
            var dense_layer = keras.layers.Dense(1);
            var outputs = dense_layer.Apply(inputs);
            var model = keras.Model(inputs, outputs);
            model.compile("adam", "mean_squared_error");
            return model;
        }
    }
}
