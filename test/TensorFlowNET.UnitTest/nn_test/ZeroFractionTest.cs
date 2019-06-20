using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Tensorflow;
using static Tensorflow.Python;

namespace TensorFlowNET.UnitTest.nn_test
{
    [TestClass]
    public class ZeroFractionTest : PythonTest
    {

        protected double _ZeroFraction(NDArray x)
        {
            assert(x.shape);
            int total_elements = np.prod(x.shape);

            var eps = 1e-8;
            var nonzeros = x.Data<double>().Count(d=>Math.Abs(d)> eps);
            return 1.0 - nonzeros / (double)total_elements;
        }

        [Ignore("TODO implement nn_impl.zero_fraction")]
        [TestMethod]
        public void testZeroFraction()
        {
            var x_shape = new Shape(5, 17);
            var x_np = np.random.randint(0, 2, x_shape);
                x_np.astype(np.float32);
            var y_np = this._ZeroFraction(x_np);

            var x_tf = constant_op.constant(x_np);
            x_tf.SetShape(x_shape);
            var y_tf = nn_impl.zero_fraction(x_tf);
            var y_tf_np = self.evaluate<NDArray>(y_tf);

            var eps = 1e-8;
            self.assertAllClose(y_tf_np, y_np, eps);
        }

        [Ignore("TODO implement nn_impl.zero_fraction")]
        [TestMethod]
        public void testZeroFractionEmpty()
        {

            var x = np.zeros(0);
            var y = self.evaluate<NDArray>(nn_impl.zero_fraction(new Tensor(x)));
            self.assertTrue(np.isnan(y));
        }

        [Ignore("TODO implement nn_impl.zero_fraction")]
        [TestMethod]
        public void testZeroFraction2_27Zeros()
        {
            var sparsity = nn_impl.zero_fraction(
                array_ops.zeros(new Shape((int) Math.Pow(2, 27 * 1.01)), dtypes.int8));
            self.assertAllClose(1.0, self.evaluate<NDArray>(sparsity));
        }

        [Ignore("TODO implement nn_impl.zero_fraction")]
        [TestMethod]
        public void testZeroFraction2_27Ones()
        {
            var sparsity = nn_impl.zero_fraction(
                array_ops.ones(new Shape((int)Math.Pow(2, 27 * 1.01)), dtypes.int8));
            self.assertAllClose(0.0, self.evaluate<NDArray>(sparsity));
        }

        [Ignore("TODO implement nn_impl.zero_fraction")]
        [TestMethod]
        public void testUnknownSize()
        {
            var value = array_ops.placeholder(dtype: dtypes.float32);
            var sparsity = nn_impl.zero_fraction(value);
            with<Session>(self.cached_session(), sess => {
                // TODO: make this compile
                      //self.assertAllClose(
                      //    0.25,
                      //    sess.run(sparsity, {value: [[0., 1.], [0.3, 2.]]}));
            });
        }


    }
}
