using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow;
using Tensorflow.UnitTest;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.Gradient
{
    [TestClass]
    public class GradientEagerTest : EagerModeTestBase
    {
        [TestMethod]
        public void ConstantSquare()
        {
            // Calcute the gradient of w * w 
            // by Automatic Differentiation in Eager mode
            // in tensorflow.net 2.x that is in development intensively
            var w = tf.constant(1.5f);
            using var tape = tf.GradientTape();
            tape.watch(w);
            var loss = w * w;
            var grad = tape.gradient(loss, w);
            Assert.AreEqual((float)grad, 3.0f);
        }

        [TestMethod]
        public void SquaredDifference_Constant()
        {
            // Calcute the gradient of (x1-x2)^2 
            // by Automatic Differentiation in Eager mode
            var x1 = tf.constant(7f);
            var x2 = tf.constant(11f);

            // Sanity check
            using (var tape = tf.GradientTape())
            {
                tape.watch(x2);
                var loss = tf.multiply((x1 - x2), (x1 - x2));

                var result = tape.gradient(loss, x2);
                // Expected is 2*(11-7) = 8
                Assert.AreEqual((float)result, 8f);
            }

            // Actual test
            using (var tape = tf.GradientTape())
            {
                tape.watch(x2);
                var loss = tf.squared_difference(x1, x2);

                // Expected is 2*(11-7) = 8
                var result = tape.gradient(loss, x2);
                Assert.AreEqual((float)result, 8f);
            }
        }


        [Ignore]
        [TestMethod]
        public void SquaredDifference_1D()
        {
            // Calcute the gradient of (x1-x2)^2 
            // by Automatic Differentiation in Eager mode
            // Expected is 2*(abs(x1-x2))
            Tensor x1 = new NumSharp.NDArray( new float[] { 1, 3, 5, 21, 19, 17 });
            Tensor x2 = new NumSharp.NDArray(new float[] { 29, 27, 23, 7, 11, 13 });
            float[] expected = new float[] {
                (29-1) * 2,
                (27-3) * 2,
                (23-5) * 2,
                (7-21) * 2,
                (11-19) * 2,
                (13-17) * 2
                };

            // Sanity check
            using (var tape = tf.GradientTape())
            {
                tape.watch(x1);
                tape.watch(x2);
                var loss = tf.multiply((x1 - x2), (x1 - x2));

                var result = tape.gradient(loss, x2);
                CollectionAssert.AreEqual(result.ToArray<float>(), expected);
            }

            // Actual test
            using (var tape = tf.GradientTape())
            {
                tape.watch(x1);
                tape.watch(x2);
                var loss = tf.squared_difference(x1, x2);

                var result = tape.gradient(loss, x2);
                CollectionAssert.AreEqual(result.ToArray<float>(), expected);
            }
        }


        /// <summary>
        /// Calcute the gradient of w * w * w
        /// 高阶梯度
        /// </summary>
        [TestMethod]
        public void HighGradient()
        {
            var x = tf.Variable(1.0f);
            using var tape1 = tf.GradientTape();
            using var tape2 = tf.GradientTape();
            var y = x * x * x;
            tape2.Dispose();
            var dy_dx = tape2.gradient(y, x);
            Assert.AreEqual((float)dy_dx, 3.0f);
            tape1.Dispose();
            var d2y_d2x = tape1.gradient(dy_dx, x);
            Assert.AreEqual((float)d2y_d2x, 6.0f);
        }

        [TestMethod]
        public void ConstantMultiply()
        {
            var x = tf.ones((2, 2));
            using var tape = tf.GradientTape();
            tape.watch(x);
            var y = tf.reduce_sum(x);
            var z = tf.multiply(y, y);
            var dz_dx = tape.gradient(z, x);

            var expected = new float[] { 8.0f, 8.0f, 8.0f, 8.0f };
            Assert.IsTrue(Enumerable.SequenceEqual(dz_dx.ToArray<float>(), expected));
        }

        [TestMethod]
        public void PersistentTape()
        {
            var x = tf.ones((2, 2));
            using var tape = tf.GradientTape(persistent: true);
            tape.watch(x);
            var y = tf.reduce_sum(x);
            var z = tf.multiply(y, y);
            tape.Dispose();

            var dz_dx = tape.gradient(z, x);

            var expected = new float[] { 8.0f, 8.0f, 8.0f, 8.0f };
            Assert.IsTrue(Enumerable.SequenceEqual(dz_dx.ToArray<float>(), expected));

            var dz_dy = tape.gradient(z, y);
            Assert.AreEqual((float)dz_dy, 8.0f);
        }

        [TestMethod]
        public void ConditionalMultiply()
        {
            Func<Tensor, int, Tensor> func = (x, y) =>
            {
                Tensor output = tf.constant(1.0f);
                foreach (var i in range(y))
                {
                    if (i > 1)
                        output = tf.multiply(output, x);
                }
                return output;
            };

            Func<Tensor, int, Tensor> grad = (x, y) =>
            {
                using var tape = tf.GradientTape();
                tape.watch(x);
                var output = func(x, y);
                var grad = tape.gradient(output, x);
                return grad;
            };

            var x = tf.constant(2.0f);
            var result = grad(x, 4);
            Assert.AreEqual((float)result, 4.0f);
        }
    }
}
