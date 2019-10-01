using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest
{
    /// <summary>
    /// Find more examples in https://www.programcreek.com/python/example/90444/tensorflow.read_file
    /// </summary>
    [TestClass]
    public class ImageTest
    {
        string imgPath = "shasta-daisy.jpg";
        Tensor contents;

        [TestInitialize]
        public void Initialize()
        {
            imgPath = Path.GetFullPath(imgPath);
            contents = tf.read_file(imgPath);
        }

        [Ignore("")]
        [TestMethod]
        public void decode_image()
        {
            var img = tf.image.decode_image(contents);
            Assert.AreEqual(img.name, "decode_image/cond_jpeg/Merge:0");
        }
    }
}
