using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System.Linq;
using Tensorflow;
using static Tensorflow.Python;

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

        [TestMethod]
        public void Gradient2x()
        {
            var graph = tf.Graph().as_default();
            using (var sess = tf.Session(graph))
            {
                var x = tf.constant(7.0f);
                var y = x * x * tf.constant(0.1f);

                var grad = tf.gradients(y, x);
                Assert.AreEqual(grad[0].name, "gradients/AddN:0");

                float r = sess.run(grad[0]);
                Assert.AreEqual(r, 1.4f);
            }
        }

        [TestMethod]
        public void Gradient3x()
        {
            var graph = tf.Graph().as_default();
            tf_with(tf.Session(graph), sess => {
                var x = tf.constant(7.0f);
                var y = x * x * x * tf.constant(0.1f);

                var grad = tf.gradients(y, x);
                Assert.AreEqual(grad[0].name, "gradients/AddN:0");

                float r = sess.run(grad[0]);
                Assert.AreEqual(r, 14.700001f);
            });
        }

        [TestMethod]
        public void StridedSlice()
        {
            var graph = tf.Graph().as_default();

            var t = tf.constant(np.array(new int[,,]
            {
                {
                    { 11, 12, 13 },
                    { 21, 22, 23 }
                },
                {
                    { 31, 32, 33 },
                    { 41, 42, 43 }
                },
                {
                    { 51, 52, 53 },
                    { 61, 62, 63 }
                }
            }));

            var slice = tf.strided_slice(t,
                begin: new[] { 0, 0, 0 },
                end: new[] { 3, 2, 3 },
                strides: new[] { 2, 2, 2 });

            var y = slice + slice;

            var g = tf.gradients(y, new Tensor[] { slice, slice });

            using (var sess = tf.Session(graph))
            {
                var r = sess.run(slice);

                Assert.IsTrue(Enumerable.SequenceEqual(r.shape, new[] { 2, 1, 2 }));
                Assert.IsTrue(Enumerable.SequenceEqual(r[0].GetData<int>(), new[] { 11, 13 }));
                Assert.IsTrue(Enumerable.SequenceEqual(r[1].GetData<int>(), new[] { 51, 53 }));
            }
        }
    }
}
