using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using static Tensorflow.Binding;

namespace TensorFlowNET.Keras.UnitTest
{
    public class EagerModeTestBase
    {
        [TestInitialize]
        public void TestInit()
        {
            if (!tf.executing_eagerly())
                tf.enable_eager_execution();
            tf.Context.ensure_initialized();
        }

        [TestCleanup]
        public void TestClean()
        {
        }

        public bool Equal(float[] f1, float[] f2)
        {
            bool ret = false;
            var tolerance = .000001f;
            for (var i = 0; i < f1.Length; i++)
            {
                ret = Math.Abs(f1[i] - f2[i]) <= tolerance;
                if (!ret)
                    break;
            }

            return ret;
        }

        public bool Equal(double[] d1, double[] d2)
        {
            bool ret = false;
            var tolerance = .000000000000001f;
            for (var i = 0; i < d1.Length; i++)
            {
                ret = Math.Abs(d1[i] - d2[i]) <= tolerance;
                if (!ret)
                    break;
            }

            return ret;
        }
    }
}
