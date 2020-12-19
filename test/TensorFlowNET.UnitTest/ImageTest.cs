using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System.Linq;
using Tensorflow;
using Tensorflow.UnitTest;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.Basics
{
    /// <summary>
    /// Find more examples in https://www.programcreek.com/python/example/90444/tensorflow.read_file
    /// </summary>
    [TestClass]
    public class ImageTest : GraphModeTestBase
    {
        string imgPath = "shasta-daisy.jpg";
        Tensor contents;

        [TestInitialize]
        public void Initialize()
        {
            imgPath = TestHelper.GetFullPathFromDataDir(imgPath);
            contents = tf.io.read_file(imgPath);
        }

        [TestMethod]
        public void decode_image()
        {
            var img = tf.image.decode_image(contents);
            Assert.AreEqual(img.name, "decode_image/Identity:0");
        }

        [TestMethod]
        public void resize_image()
        {
            tf.enable_eager_execution();
            var image = tf.constant(new int[5, 5]
            {
                {1, 0, 0, 0, 0 },
                {0, 1, 0, 0, 0 },
                {0, 0, 1, 0, 0 },
                {0, 0, 0, 1, 0 },
                {0, 0, 0, 0, 1 }
            });
            image = image[tf.newaxis, tf.ellipsis, tf.newaxis];
            image = tf.image.resize(image, (3, 5));
            image = image[0, tf.ellipsis, 0];
            Assert.IsTrue(Enumerable.SequenceEqual(new float[] { 0.6666667f, 0.3333333f, 0, 0, 0 },
                image[0].ToArray<float>()));
            Assert.IsTrue(Enumerable.SequenceEqual(new float[] { 0, 0, 1, 0, 0 },
                image[1].ToArray<float>()));
            Assert.IsTrue(Enumerable.SequenceEqual(new float[] { 0, 0, 0, 0.3333335f, 0.6666665f },
                image[2].ToArray<float>()));
            tf.compat.v1.disable_eager_execution();
        }

        [TestMethod]
        public void TestCropAndResize()
        {
            var graph = tf.Graph().as_default();

            // 3x3 'Image' with numbered coordinates
            var input = np.array(0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f);
            var image = tf.reshape(input, new int[] { 1, 3, 3, 1 });

            // 4x4 'Image' with numbered coordinates
            var input2 = np.array(0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f);
            var image2 = tf.reshape(input2, new int[] { 1, 4, 4, 1 });
            // create one box over the full image that flips it (y1 > y2)
            var box = tf.reshape(np.array(1f, 0f, 0f, 1f), new int[] { 1, 4 });
            var boxInd = tf.Variable(np.array(0));
            // crop first 3x3 imageto size 1x1
            var cropSize1_1 = tf.Variable(np.array(1, 1));
            // don't crop second 4x4 image
            var cropSize2_2 = tf.Variable(np.array(4, 4));

            var init = tf.global_variables_initializer();
            using (Session sess = tf.Session())
            {
                sess.run(init);

                var cropped = tf.image.crop_and_resize(image, box, boxInd, cropSize1_1);

                var result = sess.run(cropped);
                // check if cropped to 1x1 center was succesfull
                result.size.Should().Be(1);
                result[0, 0, 0, 0].Should().Be(4f);

                cropped = tf.image.crop_and_resize(image2, box, boxInd, cropSize2_2);
                result = sess.run(cropped);
                // check if flipped and no cropping occured
                result.size.Should().Be(16);
                result[0, 0, 0, 0].Should().Be(12f);

            }
        }
    }
}
