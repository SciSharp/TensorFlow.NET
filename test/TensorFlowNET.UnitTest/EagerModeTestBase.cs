using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest
{
    public class EagerModeTestBase : PythonTest
    {
        [TestInitialize]
        public void TestInit()
        {
            if (!tf.executing_eagerly())
                tf.enable_eager_execution();
            tf.Context.ensure_initialized();
        }

        public bool Equal(float f1, float f2)
        {
            var tolerance = .000001f;
            return Math.Abs(f1 - f2) <= tolerance;
        }

        public bool Equal(long[] l1, long[] l2)
        {
            if (l1.Length != l2.Length)
                return false;

            for (var i = 0; i < l1.Length; i++)
            {
                if (l1[i] != l2[i])
                    return false;
            }

            return true;
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
