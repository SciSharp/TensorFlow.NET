using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.ManagedAPI
{
    [TestClass]
    public class StringsApiTest
    {
        [TestMethod]
        public void StringFromBytes()
        {
            var jpg = tf.constant(new byte[] { 0x41, 0xff, 0xd8, 0xff }, tf.@string);
            var strings = jpg.ToString();
            Assert.AreEqual(strings, @"tf.Tensor: shape=(), dtype=string, numpy='A\xff\xd8\xff'");
        }

        [TestMethod]
        public void StringEqual()
        {
            var str1 = tf.constant("Hello1");
            var str2 = tf.constant("Hello2");
            var result = tf.equal(str1, str2);
            Assert.IsFalse(result.numpy());

            var str3 = tf.constant("Hello1");
            result = tf.equal(str1, str3);
            Assert.IsTrue(result.numpy());

            var str4 = tf.strings.substr(str1, 0, 5);
            var str5 = tf.strings.substr(str2, 0, 5);
            result = tf.equal(str4, str5);
            Assert.IsTrue(result.numpy());
        }

        [TestMethod]
        public void ImageType()
        {
            var imgPath = TestHelper.GetFullPathFromDataDir("shasta-daisy.jpg");
            var contents = tf.io.read_file(imgPath);

            var substr = tf.strings.substr(contents, 0, 3);
            var jpg = tf.constant(new byte[] { 0xff, 0xd8, 0xff }, tf.@string);

            var result = math_ops.equal(substr, jpg);
            Assert.IsTrue((bool)result);
        }

        [TestMethod]
        public void StringArray()
        {
            var strings = new[] { "map_and_batch_fusion", "noop_elimination", "shuffle_and_repeat_fusion" };
            var tensor = tf.constant(strings, dtype: tf.@string, name: "optimizations");
            var stringData = tensor.StringData();

            Assert.AreEqual(3, tensor.shape[0]);
            Assert.AreEqual(strings[0], stringData[0]);
            Assert.AreEqual(strings[1], stringData[1]);
            Assert.AreEqual(strings[2], stringData[2]);
        }

        [TestMethod]
        public void StringSplit()
        {
            var tensor = tf.constant(new[] { "hello world", "tensorflow .net csharp", "fsharp" });
            var ragged_tensor = tf.strings.split(tensor);
            Assert.AreEqual((3, -1), ragged_tensor.shape);
        }
    }
}
