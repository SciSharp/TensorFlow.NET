using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class GradientTest
    {
        [TestMethod]
        public void Gradients()
        {
            var graph = tf.Graph().as_default();
            var a = tf.constant(0.0);
            var b = 2.0 * a;
            Assert.AreEqual(b.name, "mul:0");
            Assert.AreEqual(b.op.inputs[0].name, "mul/x:0");
            Assert.AreEqual(b.op.inputs[1].name, "Const:0");

            var ys = a + b;
            Assert.AreEqual(ys.name, "add:0");
            Assert.AreEqual(ys.op.inputs[0].name, "Const:0");
            Assert.AreEqual(ys.op.inputs[1].name, "mul:0");

            var g = tf.gradients(ys, new Tensor[] { a, b }, stop_gradients: new Tensor[] { a, b });
            Assert.AreEqual(g[0].name, "gradients/Fill:0");
            Assert.AreEqual(g[1].name, "gradients/Fill:0");
        }
    }
}
