using Tensorflow;
using Keras.Layers;
using NumSharp;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class BaseTests
    {
        [TestMethod]
        public void Dense_Tensor_ShapeTest()
        {
            var dense_1 = new Dense(1, name: "dense_1", activation: tf.nn.relu());
            var input = new Tensor(np.array(new int[] { 3 }));
            dense_1.__build__(input.TensorShape);
            var outputShape = dense_1.output_shape(input.TensorShape);
            var a = (int[])(outputShape.dims);
            var b = (int[])(new int[] { 1 });
            var _a = np.array(a);
            var _b = np.array(b);

            Assert.IsTrue(np.array_equal(_a, _b));
        }
    }
}
