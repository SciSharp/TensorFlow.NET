using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System.Linq;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.Basics
{
    [TestClass]
    public class VariableTest : EagerModeTestBase
    {
        [TestMethod]
        public void NewVariable()
        {
            var x = tf.Variable(10, name: "x");
            Assert.AreEqual(0, x.shape.ndim);
            Assert.AreEqual(10, (int)x.numpy());
        }

        [TestMethod]
        public void StringVar()
        {
            var mammal1 = tf.Variable("Elephant", name: "var1", dtype: tf.@string);
            var mammal2 = tf.Variable("Tiger");
        }

        [TestMethod]
        public void VarSum()
        {
            var x = tf.constant(3, name: "x");
            var y = tf.Variable(x + 1, name: "y");
            Assert.AreEqual(4, (int)y.numpy());
        }

        [TestMethod]
        public void Assign1()
        {
            var variable = tf.Variable(31, name: "tree");
            var unread = variable.assign(12);
            Assert.AreEqual(12, (int)unread.numpy());
        }

        [TestMethod]
        public void Assign2()
        {
            var v1 = tf.Variable(10.0f, name: "v1");
            var v2 = v1.assign(v1 + 1.0f);
            Assert.AreEqual(v1.numpy(), v2.numpy());
            Assert.AreEqual(11f, (float)v1.numpy());
        }

        [TestMethod]
        public void Assign3()
        {
            var v1 = tf.Variable(10.0f, name: "v1");
            var v2 = tf.Variable(v1, name: "v2");
            Assert.AreEqual(v1.numpy(), v2.numpy());
            v1.assign(30.0f);
            Assert.AreNotEqual(v1.numpy(), v2.numpy());
        }

        /// <summary>
        /// Assign tensor to slice of other tensor.
        /// https://www.tensorflow.org/api_docs/python/tf/Variable#__getitem__
        /// </summary>
        [TestMethod]
        public void SliceAssign()
        {
            NDArray nd = new float[,]
            {
                { 1, 2, 3 },
                { 4, 5, 6 },
                { 7, 8, 9 }
            };
            var x = tf.Variable(nd);

            // get slice form variable
            var sliced = x[":2", ":2"];
            Assert.AreEqual(nd[0][":2"], sliced[0].numpy());
            Assert.AreEqual(nd[1][":2"], sliced[1].numpy());

            // assign to the sliced tensor
            sliced.assign(22 * tf.ones((2, 2)));

            // test assigned value
            nd = new float[,]
            {
                { 22, 22, 3 },
                { 22, 22, 6 },
                { 7, 8, 9 }
            };
            Assert.AreEqual(nd[0], x[0].numpy());
            Assert.AreEqual(nd[1], x[1].numpy());
            Assert.AreEqual(nd[2], x[2].numpy());
        }

        [TestMethod, Ignore]
        public void TypeMismatchedSliceAssign()
        {
            NDArray intNd = new int[]
            {
                1, -2, 3
            };
            NDArray doubleNd = new double[]
            {
                -5, 6, -7
            };
            var x = tf.Variable(doubleNd);

            var slice = x[":"];
            Assert.ThrowsException<System.Exception>(
                // this statement exit without throwing any exception but the "test execution summary" seems not able to detect that.
                () => slice.assign(intNd)
            );
        }

        [TestMethod]
        public void Accumulation()
        {
            var x = tf.Variable(10, name: "x");
            for (int i = 0; i < 5; i++)
                x.assign(x + 1);

            Assert.AreEqual(15, (int)x.numpy());
        }

        [TestMethod]
        public void ShouldReturnNegative()
        {
            var x = tf.constant(new[,] { { 1, 2 } });
            var neg_x = tf.negative(x);
            Assert.IsTrue(Enumerable.SequenceEqual(new[] { 1, 2 }, neg_x.shape));
            Assert.IsTrue(Enumerable.SequenceEqual(new[] { -1, -2 }, neg_x.numpy().ToArray<int>()));
        }

        [TestMethod]
        public void IdentityOriginalTensor()
        {
            var a = tf.Variable(5);
            var a_identity = tf.identity(a);
            a.assign_add(1);
            Assert.AreEqual(5, (int)a_identity.numpy());
            Assert.AreEqual(6, (int)a.numpy());
        }
    }
}
