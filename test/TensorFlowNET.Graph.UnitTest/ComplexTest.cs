using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow.NumPy;
using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow;
using static Tensorflow.Binding;
using Tensorflow.Keras.UnitTest;

namespace TensorFlowNET.UnitTest.Basics
{
    [TestClass]
    public class ComplexTest : EagerModeTestBase
    {
        // Tests for Complex128

        [TestMethod]
        public void complex128_basic()
        {
            double[] d_real = new double[] { 1.0, 2.0, 3.0, 4.0 };
            double[] d_imag = new double[] { -1.0, -3.0, 5.0, 7.0 };

            Tensor t_real = tf.constant(d_real, dtype:TF_DataType.TF_DOUBLE);
            Tensor t_imag = tf.constant(d_imag, dtype: TF_DataType.TF_DOUBLE);

            Tensor t_complex = tf.complex(t_real, t_imag);

            Tensor t_real_result = tf.math.real(t_complex);
            Tensor t_imag_result = tf.math.imag(t_complex);

            NDArray n_real_result = t_real_result.numpy();
            NDArray n_imag_result = t_imag_result.numpy();

            double[] d_real_result =n_real_result.ToArray<double>();
            double[] d_imag_result = n_imag_result.ToArray<double>();

            Assert.IsTrue(base.Equal(d_real_result, d_real));
            Assert.IsTrue(base.Equal(d_imag_result, d_imag));
        }
        [TestMethod]
        public void complex128_abs()
        {
            tf.enable_eager_execution();

            double[] d_real = new double[] { -3.0, -5.0, 8.0, 7.0 };
            double[] d_imag = new double[] { -4.0, 12.0, -15.0, 24.0 };

            double[] d_abs = new double[] { 5.0, 13.0, 17.0, 25.0 };

            Tensor t_real = tf.constant(d_real, dtype: TF_DataType.TF_DOUBLE);
            Tensor t_imag = tf.constant(d_imag, dtype: TF_DataType.TF_DOUBLE);

            Tensor t_complex = tf.complex(t_real, t_imag);

            Tensor t_abs_result = tf.abs(t_complex);

            double[] d_abs_result = t_abs_result.numpy().ToArray<double>();
            Assert.IsTrue(base.Equal(d_abs_result, d_abs));
        }
        [TestMethod]
        public void complex128_conj()
        {
            double[] d_real = new double[] { -3.0, -5.0, 8.0, 7.0 };
            double[] d_imag = new double[] { -4.0, 12.0, -15.0, 24.0 };

            double[] d_real_expected = new double[] { -3.0, -5.0, 8.0, 7.0 };
            double[] d_imag_expected = new double[] { 4.0, -12.0, 15.0, -24.0 };

            Tensor t_real = tf.constant(d_real, dtype: TF_DataType.TF_DOUBLE);
            Tensor t_imag = tf.constant(d_imag, dtype: TF_DataType.TF_DOUBLE);

            Tensor t_complex = tf.complex(t_real, t_imag, TF_DataType.TF_COMPLEX128);

            Tensor t_result = tf.math.conj(t_complex);

            NDArray n_real_result = tf.math.real(t_result).numpy();
            NDArray n_imag_result = tf.math.imag(t_result).numpy();

            double[] d_real_result = n_real_result.ToArray<double>();
            double[] d_imag_result = n_imag_result.ToArray<double>();

            Assert.IsTrue(base.Equal(d_real_result, d_real_expected));
            Assert.IsTrue(base.Equal(d_imag_result, d_imag_expected));
        }
        [TestMethod]
        public void complex128_angle()
        {
            double[] d_real = new double[] { 0.0, 1.0, -1.0, 0.0 };
            double[] d_imag = new double[] { 1.0, 0.0, -2.0, -3.0 };

            double[] d_expected = new double[] { 1.5707963267948966, 0, -2.0344439357957027, -1.5707963267948966 };

            Tensor t_real = tf.constant(d_real, dtype: TF_DataType.TF_DOUBLE);
            Tensor t_imag = tf.constant(d_imag, dtype: TF_DataType.TF_DOUBLE);

            Tensor t_complex = tf.complex(t_real, t_imag, TF_DataType.TF_COMPLEX128);

            Tensor t_result = tf.math.angle(t_complex);

            NDArray n_result = t_result.numpy();

            double[] d_result = n_result.ToArray<double>();

            Assert.IsTrue(base.Equal(d_result, d_expected));
        }

        // Tests for Complex64
        [TestMethod]
        public void complex64_basic()
        {
            tf.init_scope();
            float[] d_real = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            float[] d_imag = new float[] { -1.0f, -3.0f, 5.0f, 7.0f };

            Tensor t_real = tf.constant(d_real, dtype: TF_DataType.TF_FLOAT);
            Tensor t_imag = tf.constant(d_imag, dtype: TF_DataType.TF_FLOAT);

            Tensor t_complex = tf.complex(t_real, t_imag);

            Tensor t_real_result = tf.math.real(t_complex);
            Tensor t_imag_result = tf.math.imag(t_complex);

            // Convert the EagerTensors to NumPy arrays directly
            float[] d_real_result = t_real_result.numpy().ToArray<float>();
            float[] d_imag_result = t_imag_result.numpy().ToArray<float>();

            Assert.IsTrue(base.Equal(d_real_result, d_real));
            Assert.IsTrue(base.Equal(d_imag_result, d_imag));
        }
        [TestMethod]
        public void complex64_abs()
        {
            tf.enable_eager_execution();
            
            float[] d_real = new float[] { -3.0f, -5.0f, 8.0f, 7.0f };
            float[] d_imag = new float[] { -4.0f, 12.0f, -15.0f, 24.0f };

            float[] d_abs = new float[] { 5.0f, 13.0f, 17.0f, 25.0f };

            Tensor t_real = tf.constant(d_real, dtype: TF_DataType.TF_FLOAT);
            Tensor t_imag = tf.constant(d_imag, dtype: TF_DataType.TF_FLOAT);

            Tensor t_complex = tf.complex(t_real, t_imag, TF_DataType.TF_COMPLEX64);

            Tensor t_abs_result = tf.abs(t_complex);

            NDArray n_abs_result = t_abs_result.numpy();

            float[] d_abs_result = n_abs_result.ToArray<float>();
            Assert.IsTrue(base.Equal(d_abs_result, d_abs));

        }
        [TestMethod]
        public void complex64_conj()
        {
            float[] d_real = new float[] { -3.0f, -5.0f, 8.0f, 7.0f };
            float[] d_imag = new float[] { -4.0f, 12.0f, -15.0f, 24.0f };

            float[] d_real_expected = new float[] { -3.0f, -5.0f, 8.0f, 7.0f };
            float[] d_imag_expected = new float[] { 4.0f, -12.0f, 15.0f, -24.0f };

            Tensor t_real = tf.constant(d_real, dtype: TF_DataType.TF_FLOAT);
            Tensor t_imag = tf.constant(d_imag, dtype: TF_DataType.TF_FLOAT);

            Tensor t_complex = tf.complex(t_real, t_imag, TF_DataType.TF_COMPLEX64);

            Tensor t_result = tf.math.conj(t_complex);

            NDArray n_real_result = tf.math.real(t_result).numpy();
            NDArray n_imag_result = tf.math.imag(t_result).numpy();

            float[] d_real_result = n_real_result.ToArray<float>();
            float[] d_imag_result = n_imag_result.ToArray<float>();

            Assert.IsTrue(base.Equal(d_real_result, d_real_expected));
            Assert.IsTrue(base.Equal(d_imag_result, d_imag_expected));

        }
        [TestMethod]
        public void complex64_angle()
        {
            float[] d_real = new float[] { 0.0f, 1.0f, -1.0f, 0.0f };
            float[] d_imag = new float[] { 1.0f, 0.0f, -2.0f, -3.0f };

            float[] d_expected = new float[] { 1.5707964f, 0f, -2.0344439f, -1.5707964f };
            
            Tensor t_real = tf.constant(d_real, dtype: TF_DataType.TF_FLOAT);
            Tensor t_imag = tf.constant(d_imag, dtype: TF_DataType.TF_FLOAT);

            Tensor t_complex = tf.complex(t_real, t_imag, TF_DataType.TF_COMPLEX64);

            Tensor t_result = tf.math.angle(t_complex);

            NDArray n_result = t_result.numpy();

            float[] d_result = n_result.ToArray<float>();

            Assert.IsTrue(base.Equal(d_result, d_expected));
        }
    }
}