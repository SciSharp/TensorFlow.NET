using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.ManagedAPI
{
    [TestClass]
    public class LinalgTest : EagerModeTestBase
    {
        [TestMethod]
        public void EyeTest()
        {
            var tensor = tf.linalg.eye(3);

            Assert.AreEqual(tensor.shape, (3, 3));

            Assert.AreEqual(0.0f, (double)tensor[2, 0]);
            Assert.AreEqual(0.0f, (double)tensor[2, 1]);
            Assert.AreEqual(1.0f, (double)tensor[2, 2]);
        }

        /// <summary>
        /// https://colab.research.google.com/github/biswajitsahoo1111/blog_notebooks/blob/master/Doing_Linear_Algebra_using_Tensorflow_2.ipynb#scrollTo=6xfOcTFBL3Up
        /// </summary>
        [TestMethod]
        public void LSTSQ()
        {
            var A_over = tf.constant(new float[,] { { 1, 2 }, { 2, 0.5f }, { 3, 1 }, { 4, 5.0f} });
            var A_under = tf.constant(new float[,] { { 3, 1, 2, 5 }, { 7, 9, 1, 4.0f } });
            var b_over = tf.constant(new float[] { 3, 4, 5, 6.0f }, shape: (4, 1));
            var b_under = tf.constant(new float[] { 7.2f, -5.8f }, shape: (2, 1));
            var x_over = tf.linalg.lstsq(A_over, b_over);

            var x = tf.matmul(tf.linalg.inv(tf.matmul(A_over, A_over, transpose_a: true)), tf.matmul(A_over, b_over, transpose_a: true));
            Assert.AreEqual(x_over.shape, (2, 1));
            AssetSequenceEqual(x_over.ToArray<float>(), x.ToArray<float>());

            var x_under = tf.linalg.lstsq(A_under, b_under);
            var y = tf.matmul(A_under, tf.matmul(tf.linalg.inv(tf.matmul(A_under, A_under, transpose_b: true)), b_under), transpose_a: true);

            Assert.AreEqual(x_under.shape, (4, 1));
            AssetSequenceEqual(x_under.ToArray<float>(), y.ToArray<float>());

            /*var x_over_reg = tf.linalg.lstsq(A_over, b_over, l2_regularizer: 2.0f);
            var x_under_reg = tf.linalg.lstsq(A_under, b_under, l2_regularizer: 2.0f);
            Assert.AreEqual(x_under_reg.shape, (4, 1));
            AssetSequenceEqual(x_under_reg.ToArray<float>(), new float[] { -0.04763567f, -1.214508f, 0.62748903f, 1.299031f });*/
        }

        [TestMethod]
        public void Einsum()
        {
            var m0 = tf.random.normal((2, 3));
            var m1 = tf.random.normal((3, 5));
            var e = tf.linalg.einsum("ij,jk->ik", (m0, m1));
            Assert.AreEqual(e.shape, (2, 5));
        }

        [TestMethod]
        public void GlobalNorm()
        {
            var t_list = new Tensors(tf.constant(new float[] { 1, 2, 3, 4 }), tf.constant(new float[] { 5, 6, 7, 8 }));
            var norm = tf.linalg.global_norm(t_list);
            Assert.AreEqual(norm.numpy(), 14.282857f);
        }

        [TestMethod]
        public void Tensordot()
        {
            var a = tf.constant(new[] { 1, 2 });
            var b = tf.constant(new[] { 2, 3 });
            var c = tf.linalg.tensordot(a, b, 0);
            Assert.AreEqual(c.shape, (2, 2));
            AssetSequenceEqual(c.ToArray<int>(), new[] { 2, 3, 4, 6 });

            c = tf.linalg.tensordot(a, b, new[] { 0, 0 });
            Assert.AreEqual(c.shape.ndim, 0);
            Assert.AreEqual(c.numpy(), 8);
        }

        [TestMethod]
        public void Matmul()
        {
            var a = tf.constant(new[] { 1, 2, 3, 4, 5, 6 }, shape: (2, 3));
            var b = tf.constant(new[] { 7, 8, 9, 10, 11, 12 }, shape: (3, 2));
            var c = tf.linalg.matmul(a, b);

            Assert.AreEqual(c.shape, (2, 2));
            AssetSequenceEqual(c.ToArray<int>(), new[] { 58, 64, 139, 154 });
        }
    }
}
