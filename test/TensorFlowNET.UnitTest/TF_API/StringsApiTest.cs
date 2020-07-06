using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.UnitTest.TF_API
{
    [TestClass]
    public class StringsApiTest
    {
        [TestMethod]
        public void StringEqual()
        {
            var str1 = tf.constant("Hello1");
            var str2 = tf.constant("Hello2");
            var result = tf.equal(str1, str2);
            Assert.IsFalse(result.ToScalar<bool>());

            var str3 = tf.constant("Hello1");
            result = tf.equal(str1, str3);
            Assert.IsTrue(result.ToScalar<bool>());

            var str4 = tf.strings.substr(str1, 0, 5);
            var str5 = tf.strings.substr(str2, 0, 5);
            result = tf.equal(str4, str5);
            Assert.IsTrue(result.ToScalar<bool>());
        }

        [TestMethod]
        public void ImageType()
        {
            var imgPath = TestHelper.GetFullPathFromDataDir("shasta-daisy.jpg");
            var contents = tf.io.read_file(imgPath);

            var substr = tf.strings.substr(contents, 0, 3);
            var jpg = Encoding.UTF8.GetString(new byte[] { 0xff, 0xd8, 0xff });
            var jpg_tensor = tf.constant(jpg);

            var result = math_ops.equal(substr, jpg_tensor);
        }
    }
}
