using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class ImageTest
    {
        string imgPath = "../../../../../data/shasta-daisy.jpg";
        Tensor contents;

        public ImageTest()
        {
            imgPath = Path.GetFullPath(imgPath);
            contents = tf.read_file(imgPath);
        }

        [TestMethod]
        public void decode_image()
        {
            var img = tf.image.decode_image(contents);
            Assert.AreEqual(img.name, "decode_image/cond_jpeg/Merge:0");
        }
    }
}
