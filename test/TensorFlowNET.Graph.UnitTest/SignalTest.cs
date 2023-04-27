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
    public class SignalTest : EagerModeTestBase
    {
        [TestMethod]
        public void fft()
        {
            double[] d_real = new double[] { 1.0, 2.0, 3.0, 4.0 };
            double[] d_imag = new double[] { -1.0, -3.0, 5.0, 7.0 };

            Tensor t_real = tf.constant(d_real, dtype: TF_DataType.TF_DOUBLE);
            Tensor t_imag = tf.constant(d_imag, dtype: TF_DataType.TF_DOUBLE);

            Tensor t_complex = tf.complex(t_real, t_imag);

            Tensor t_frequency_domain = tf.signal.fft(t_complex);
            Tensor f_time_domain = tf.signal.ifft(t_frequency_domain);

            Tensor t_real_result = tf.math.real(f_time_domain);
            Tensor t_imag_result = tf.math.imag(f_time_domain);

            NDArray n_real_result = t_real_result.numpy();
            NDArray n_imag_result = t_imag_result.numpy();

            double[] d_real_result = n_real_result.ToArray<double>();
            double[] d_imag_result = n_imag_result.ToArray<double>();

            Assert.IsTrue(base.Equal(d_real_result, d_real));
            Assert.IsTrue(base.Equal(d_imag_result, d_imag));
        }
        [TestMethod]
        public void fft2d()
        {
            double[] d_real = new double[] { 1.0, 2.0, 3.0, 4.0 };
            double[] d_imag = new double[] { -1.0, -3.0, 5.0, 7.0 };

            Tensor t_real = tf.constant(d_real, dtype: TF_DataType.TF_DOUBLE);
            Tensor t_imag = tf.constant(d_imag, dtype: TF_DataType.TF_DOUBLE);

            Tensor t_complex = tf.complex(t_real, t_imag);

            Tensor t_complex_2d = tf.reshape(t_complex,new int[] { 2, 2 });

            Tensor t_frequency_domain_2d = tf.signal.fft2d(t_complex_2d);
            Tensor t_time_domain_2d = tf.signal.ifft2d(t_frequency_domain_2d);

            Tensor t_time_domain = tf.reshape(t_time_domain_2d, new int[] { 4 });

            Tensor t_real_result = tf.math.real(t_time_domain);
            Tensor t_imag_result = tf.math.imag(t_time_domain);

            NDArray n_real_result = t_real_result.numpy();
            NDArray n_imag_result = t_imag_result.numpy();

            double[] d_real_result = n_real_result.ToArray<double>();
            double[] d_imag_result = n_imag_result.ToArray<double>();

            Assert.IsTrue(base.Equal(d_real_result, d_real));
            Assert.IsTrue(base.Equal(d_imag_result, d_imag));
        }
        [TestMethod]
        public void fft3d()
        {
            double[] d_real = new double[] { 1.0, 2.0, 3.0, 4.0, -3.0, -2.0, -1.0, -4.0 };
            double[] d_imag = new double[] { -1.0, -3.0, 5.0, 7.0, 6.0, 4.0, 2.0, 0.0};

            Tensor t_real = tf.constant(d_real, dtype: TF_DataType.TF_DOUBLE);
            Tensor t_imag = tf.constant(d_imag, dtype: TF_DataType.TF_DOUBLE);

            Tensor t_complex = tf.complex(t_real, t_imag);

            Tensor t_complex_3d = tf.reshape(t_complex, new int[] { 2, 2, 2 });

            Tensor t_frequency_domain_3d = tf.signal.fft2d(t_complex_3d);
            Tensor t_time_domain_3d = tf.signal.ifft2d(t_frequency_domain_3d);

            Tensor t_time_domain = tf.reshape(t_time_domain_3d, new int[] { 8 });

            Tensor t_real_result = tf.math.real(t_time_domain);
            Tensor t_imag_result = tf.math.imag(t_time_domain);

            NDArray n_real_result = t_real_result.numpy();
            NDArray n_imag_result = t_imag_result.numpy();

            double[] d_real_result = n_real_result.ToArray<double>();
            double[] d_imag_result = n_imag_result.ToArray<double>();

            Assert.IsTrue(base.Equal(d_real_result, d_real));
            Assert.IsTrue(base.Equal(d_imag_result, d_imag));
        }
    }
}