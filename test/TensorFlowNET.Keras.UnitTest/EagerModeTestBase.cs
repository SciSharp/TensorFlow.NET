using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.UnitTest
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


        public void AssertArray(int[] f1, int[] f2)
        {
            bool ret = false;
            for (var i = 0; i < f1.Length; i++)
            {
                ret = f1[i] == f2[i];
                if (!ret)
                    break;
            }

            if (!ret)
            {
                Assert.Fail($"Array not Equal:[{string.Join(",", f1)}] [{string.Join(",", f2)}]");
            }
        }

        public void AssertArray(float[] f1, float[] f2)
        {
            bool ret = false;
            var tolerance = .00001f;
            for (var i = 0; i < f1.Length; i++)
            {
                ret = Math.Abs(f1[i] - f2[i]) <= tolerance;
                if (!ret)
                    break;
            }

            if (!ret)
            {
                Assert.Fail($"Array float not Equal:[{string.Join(",", f1)}] [{string.Join(",", f2)}]");
            }
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
