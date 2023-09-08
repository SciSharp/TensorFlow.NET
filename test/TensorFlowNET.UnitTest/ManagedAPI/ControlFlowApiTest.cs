using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow.NumPy;
using System;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.ManagedAPI
{
    [TestClass]
    public class ControlFlowApiTest
    {
        [TestMethod]
        public void WhileLoopOneInputEagerMode()
        {
            tf.enable_eager_execution();

            var i = tf.constant(2);
            Func<Tensor, Tensor> c = (x) => tf.less(x, 10);
            Func<Tensor, Tensor> b = (x) => tf.add(x, 1);
            var r = tf.while_loop(c, b, i);
            Assert.AreEqual(10, (int)r);
        }

        [TestMethod]
        public void WhileLoopTwoInputsEagerMode()
        {
            tf.enable_eager_execution();

            var i = tf.constant(2);
            var j = tf.constant(3);
            Func<Tensors, Tensor> c = (x) => tf.less(x[0] + x[1], 10);
            Func<Tensors, Tensors> b = (x) => new[] { tf.add(x[0], 1), tf.add(x[1], 1) };
            var r = tf.while_loop(c, b, new[] { i, j });
            Assert.AreEqual(5, (int)r[0]);
            Assert.AreEqual(6, (int)r[1]);
        }

        [TestMethod, Ignore]
        public void WhileLoopGraphMode()
        {
            tf.compat.v1.disable_eager_execution();

            var i = tf.constant(2);
            Func<Tensor, Tensor> c = (x) => tf.less(x, 10);
            Func<Tensor, Tensor> b = (x) => tf.add(x, 1);
            var r = tf.while_loop(c, b, i);
            Assert.AreEqual(10, (int)r);
        }


        [TestMethod, Ignore]
        public void ScanFunctionGraphMode()
        {
            tf.compat.v1.disable_eager_execution();

            Func<Tensor, Tensor, Tensor> fn = (prev, current) => tf.add(prev, current);
            var input = tf.placeholder(TF_DataType.TF_FLOAT, new Shape(6));
            var scan = tf.scan(fn, input);

            var sess = tf.Session();
            sess.run(tf.global_variables_initializer());
            var result = sess.run(scan, new FeedItem(input, np.array(1, 2, 3, 4, 5, 6)));
            Assert.AreEqual(new float[] { 1, 3, 6, 10, 15, 21 }, result.ToArray<float>());
        }
    }
}
