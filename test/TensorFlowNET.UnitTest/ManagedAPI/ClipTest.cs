using Microsoft.VisualStudio.TestTools.UnitTesting;
using static Tensorflow.Binding;
using Tensorflow;

namespace TensorFlowNET.UnitTest.ClipOps
{
    [TestClass]
    public class ClipTest : EagerModeTestBase
    {
        [TestMethod]
        public void clip_by_global_norm()
        {
            var t_list = new Tensors(tf.constant(new float[] { 1, 2, 3, 4 }), tf.constant(new float[] { 5, 6, 7, 8 }));
            var clip_norm = .8f;
            var (res, norm) = tf.clip_by_global_norm(t_list, clip_norm);
            Equal(res[0].ToArray<float>(), new[] { 0.0560112074f, 0.112022415f, 0.16803363f, 0.22404483f });
            Equal(res[1].ToArray<float>(), new[] { 0.28005603f, 0.336067259f, 0.392078459f, 0.448089659f });
            Assert.AreEqual(norm.numpy(), 14.282857f);
        }
    }
}
