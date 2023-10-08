using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow.NumPy;
using System.Linq;
using Tensorflow;
using static Tensorflow.Binding;
using System;
using System.IO;

namespace TensorFlowNET.UnitTest
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
        public void adjust_contrast()
        {
            var input = np.array(0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f);
            var image = tf.reshape(input, new int[] { 3, 3, 1 });

            var init = tf.global_variables_initializer();
            var sess = tf.Session();
            sess.run(init);
            var adjust_contrast = tf.image.adjust_contrast(image, 2.0f);
            var result = sess.run(adjust_contrast);
            var res = np.array(-4f, -2f, 0f, 2f, 4f, 6f, 8f, 10f, 12f).reshape((3,3,1));
            Assert.AreEqual(result.numpy(), res);
        }

        [Ignore]
        [TestMethod]
        public void adjust_hue()
        {
            var image = tf.constant(new int[] {1,2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18});
            image = tf.reshape(image, new int[] { 3, 2, 3 });
            var adjusted_image = tf.image.adjust_hue(image, 0.2f);
            var res = tf.constant(new int[] {2,1,3, 4, 5, 6,8,7,9,11,10,12,14,13,15,17,16,18});
            res = tf.reshape(res,(3,2,3));
            Assert.AreEqual(adjusted_image, res);
        }

        [TestMethod]
        public void combined_non_max_suppression()
        {
            var boxesX = tf.constant(new float[,] { { 200, 100, 150, 100 }, { 220, 120, 150, 100 }, { 190, 110, 150, 100 }, { 210, 112, 150, 100 } });
            var boxes1 = tf.reshape(boxesX, (1, 4, 1, 4));
            var scoresX = tf.constant(new float[,] { { 0.2f, 0.7f, 0.1f }, { 0.1f, 0.8f, 0.1f }, { 0.3f, 0.6f, 0.1f }, { 0.05f, 0.9f, 0.05f } });
            var scores1 = tf.reshape(scoresX, (1, 4, 3));

            var init = tf.global_variables_initializer();
            var sess = tf.Session();
            sess.run(init);

            var (boxes, scores, classes, valid_detections) = tf.image.combined_non_max_suppression(boxes1, scores1, 10, 10, 0.5f, 0.2f, clip_boxes: false);
            var result = sess.run((boxes, scores, classes, valid_detections));

            var boxes_gt = tf.constant(new float[,] { { 210f, 112f, 150f, 100f }, { 200f, 100f, 150f, 100f }, { 190f, 110f, 150f, 100f },
                { 0f, 0f, 0f, 0f},{ 0f, 0f, 0f, 0f},{ 0f, 0f, 0f, 0f},{ 0f, 0f, 0f , 0f},{ 0f, 0f, 0f, 0f},{ 0f , 0f, 0f, 0f},{ 0f, 0f, 0f, 0f} });
            boxes_gt = tf.reshape(boxes_gt, (1, 10, 4));
            Assert.AreEqual(result.Item1.numpy(), boxes_gt.numpy());
            var scores_gt = tf.constant(new float[,] { { 0.9f, 0.7f, 0.3f, 0f, 0f, 0f, 0f, 0f, 0f, 0f } });
            scores_gt = tf.reshape(scores_gt, (1, 10));
            Assert.AreEqual(result.Item2.numpy(), scores_gt.numpy());
            var classes_gt = tf.constant(new float[,] { { 1f, 1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f } });
            classes_gt = tf.reshape(classes_gt, (1, 10));
            Assert.AreEqual(result.Item3.numpy(), classes_gt.numpy());
            var valid_detections_gt = tf.constant(new int[,] { { 3 } });
            valid_detections_gt = tf.reshape(valid_detections_gt, (1));
            Assert.AreEqual(result.Item4.numpy(), valid_detections_gt.numpy());
        }

        [TestMethod]
        public void crop_and_resize()
        {
            int BATCH_SIZE = 1;
            int NUM_BOXES = 5;
            int IMAGE_HEIGHT = 256;
            int IMAGE_WIDTH = 256;
            int CHANNELS = 3;
            var crop_size = tf.constant(new int[] { 24, 24 });
            var image = tf.random.uniform((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS));
            var boxes = tf.random.uniform((NUM_BOXES, 4));
            var box_ind = tf.random.uniform((NUM_BOXES), minval: 0, maxval: BATCH_SIZE, dtype: TF_DataType.TF_INT32);
            var output = tf.image.crop_and_resize(image, boxes, box_ind, crop_size);
            Assert.AreEqual((5,24,24,3), output.shape);
        }

        [TestMethod]
        public void decode_image()
        {
            var img = tf.image.decode_image(contents);
            Assert.AreEqual(img.name, "decode_image/DecodeImage:0");
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
            var sess = tf.Session();
            sess.run(init);

            var cropped = tf.image.crop_and_resize(image, box, boxInd, cropSize1_1);

            var result = sess.run(cropped);
            // check if cropped to 1x1 center was succesfull
            Assert.AreEqual(result.size, 1ul);
            Assert.AreEqual(result[0, 0, 0, 0], 4f);

            cropped = tf.image.crop_and_resize(image2, box, boxInd, cropSize2_2);
            result = sess.run(cropped);
            // check if flipped and no cropping occured
            Assert.AreEqual(result.size, 16ul);
            Assert.AreEqual(result[0, 0, 0, 0], 12f);
        }

        [TestMethod]
        public void ImageSaveTest()
        {
            var imgPath = TestHelper.GetFullPathFromDataDir("img001.bmp");
            var jpegImgPath = TestHelper.GetFullPathFromDataDir("img001.jpeg");
            var pngImgPath = TestHelper.GetFullPathFromDataDir("img001.png");

            File.Delete(jpegImgPath);
            File.Delete(pngImgPath);

            var contents = tf.io.read_file(imgPath);
            var bmp = tf.image.decode_image(contents);
            Assert.AreEqual(bmp.name, "decode_image/DecodeImage:0");

            var jpeg = tf.image.encode_jpeg(bmp);
            var op1 = tf.io.write_file(jpegImgPath, jpeg);

            var png = tf.image.encode_png(bmp);
            var op2 = tf.io.write_file(pngImgPath, png);

            this.session().run(op1);
            this.session().run(op2);

            Assert.IsTrue(File.Exists(jpegImgPath), "not find file:" + jpegImgPath);
            Assert.IsTrue(File.Exists(pngImgPath), "not find file:" + pngImgPath);

            // 如果要测试图片正确性，需要注释下面两行代码
            File.Delete(jpegImgPath);
            File.Delete(pngImgPath);
        }

        [TestMethod]
        public void ImageFlipTest()
        {
            var imgPath = TestHelper.GetFullPathFromDataDir("img001.bmp");

            var contents = tf.io.read_file(imgPath);
            var bmp = tf.image.decode_image(contents);

            // 左右翻转
            var lrImgPath = TestHelper.GetFullPathFromDataDir("img001_lr.png");
            File.Delete(lrImgPath);

            var lr = tf.image.flip_left_right(bmp);
            var png = tf.image.encode_png(lr);
            var op = tf.io.write_file(lrImgPath, png);
            this.session().run(op);

            Assert.IsTrue(File.Exists(lrImgPath), "not find file:" + lrImgPath);

            // 上下翻转
            var updownImgPath = TestHelper.GetFullPathFromDataDir("img001_updown.png");
            File.Delete(updownImgPath);

            var updown = tf.image.flip_up_down(bmp);
            var pngupdown = tf.image.encode_png(updown);
            var op2 = tf.io.write_file(updownImgPath, pngupdown);
            this.session().run(op2);
            Assert.IsTrue(File.Exists(updownImgPath));


            // 暂时先人工观测图片是否翻转，观测时需要删除下面这两行代码
            File.Delete(lrImgPath);
            File.Delete(updownImgPath);

            // 多图翻转
            // 目前直接通过 bmp 拿到 shape ，这里先用默认定义图片大小来构建了
            var mImg = tf.stack(new[] { bmp, lr }, axis:0);
            print(mImg.shape);

            var up2 = tf.image.flip_up_down(mImg);

            var updownImgPath_m1 = TestHelper.GetFullPathFromDataDir("img001_m_ud.png");   // 直接上下翻转
            File.Delete(updownImgPath_m1);

            var img001_updown_m2 = TestHelper.GetFullPathFromDataDir("img001_m_lr_ud.png");   // 先左右再上下
            File.Delete(img001_updown_m2);

            var png2 = tf.image.encode_png(up2[0]);
            tf.io.write_file(updownImgPath_m1, png2);

            png2 = tf.image.encode_png(up2[1]);
            tf.io.write_file(img001_updown_m2, png2);

            // 如果要测试图片正确性，需要注释下面两行代码
            File.Delete(updownImgPath_m1);
            File.Delete(img001_updown_m2);
        }
    }
}
