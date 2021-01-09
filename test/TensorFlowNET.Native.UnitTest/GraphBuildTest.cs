using Microsoft.VisualStudio.TestTools.UnitTesting;
using static Tensorflow.Binding;

namespace Tensorflow.Native.UnitTest
{
    [TestClass]
    public class GraphBuildTest : CApiTest
    {
        [TestMethod, Ignore("Waiting to merge https://github.com/tensorflow/tensorflow/pull/43383")]
        public void UpdateEdge()
        {
            using var graph = new Graph().as_default();

            var one = tf.constant(1, name: "one");
            var two = tf.constant(2, name: "two");
            var add = tf.add(one, two, name: "add");
            var neg = tf.negative(add, name: "neg");

            Assert.AreEqual(1, one.consumers().Length);
            Assert.AreEqual("add", neg.op.node_def.Input[0]);

            // update edge
            neg.op._update_input(0, one);
            // c_api.TF_UpdateEdge(graph, new TF_Output(c1.op, 0), new TF_Input(neg.op, 0), tf.Status.Handle);

            Assert.AreEqual(2, one.consumers().Length);
            Assert.AreEqual("one:0", neg.op.node_def.Input[0]);
        }
    }
}
