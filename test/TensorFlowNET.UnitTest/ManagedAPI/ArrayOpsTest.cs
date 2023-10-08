using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow.NumPy;
using Tensorflow;
using static Tensorflow.Binding;
using System.Linq;
using Tensorflow.Operations;

namespace TensorFlowNET.UnitTest.ManagedAPI
{
    [TestClass]
    public class ArrayOpsTest : EagerModeTestBase
    {
        /// <summary>
        /// https://www.tensorflow.org/api_docs/python/tf/slice
        /// </summary>
        [TestMethod]
        public void Slice()
        {
            // Tests based on example code in TF documentation
            var input_array = tf.constant(np.array(new int[] { 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6 }).reshape((3,2,3)));
            var indices = tf.constant(np.array(new int[] { 0, 2 }));

            var r1 = array_ops.slice(input_array, ops.convert_n_to_tensor(new object[] { 1, 0, 0 }), ops.convert_n_to_tensor(new object[] { 1, 1, 3 }));
            Assert.AreEqual(new Shape(1,1,3), r1.shape);
            var r1np = r1.numpy();
            Assert.AreEqual(r1np[0, 0, 0], 3);
            Assert.AreEqual(r1np[0, 0, 1], 3);
            Assert.AreEqual(r1np[0, 0, 2], 3);


            var r2 = array_ops.slice(input_array, ops.convert_n_to_tensor(new object[] { 1, 0, 0 }), ops.convert_n_to_tensor(new object[] { 1, 2, 3 }));
            Assert.AreEqual(new Shape(1, 2, 3), r2.shape);
            var r2np = r2.numpy();
            Assert.AreEqual(r2np[0, 0, 0], 3);
            Assert.AreEqual(r2np[0, 0, 1], 3);
            Assert.AreEqual(r2np[0, 0, 2], 3);
            Assert.AreEqual(r2np[0, 1, 0], 4);
            Assert.AreEqual(r2np[0, 1, 1], 4);
            Assert.AreEqual(r2np[0, 1, 2], 4);

            var r3 = array_ops.slice(input_array, ops.convert_n_to_tensor(new object[] { 1, 0, 0 }), ops.convert_n_to_tensor(new object[] { 2, 1, 3 }));
            Assert.AreEqual(new Shape(2, 1, 3), r3.shape);
            var r3np = r3.numpy();
            Assert.AreEqual(r3np[0, 0, 0], 3);
            Assert.AreEqual(r3np[0, 0, 1], 3);
            Assert.AreEqual(r3np[0, 0, 2], 3);
            Assert.AreEqual(r3np[1, 0, 0], 5);
            Assert.AreEqual(r3np[1, 0, 1], 5);
            Assert.AreEqual(r3np[1, 0, 2], 5);
        }

        /// <summary>
        /// https://www.tensorflow.org/api_docs/python/tf/gather
        /// </summary>
        [TestMethod]
        public void Gather()
        {
            var input_array = tf.constant(np.arange(12).reshape((3, 4)).astype(np.float32));
            var indices = tf.constant(np.array(new int[] { 0, 2 }));

            var result = array_ops.gather(input_array, indices);
            Assert.AreEqual(new Shape(2, 4), result.shape);
            Assert.AreEqual(result.numpy()[0, 0], 0.0f);
            Assert.AreEqual(result.numpy()[0, 1], 1.0f);
            Assert.AreEqual(result.numpy()[1, 3], 11.0f);

            // Tests based on example code in Python doc string for tf.gather()

            var p1 = tf.random.normal(new Shape(5, 6, 7, 8));
            var i1 = tf.random_uniform(new Shape(10, 11), maxval: 7, dtype: tf.int32);
            var r1 = tf.gather(p1, i1, axis:2);
            Assert.AreEqual(new Shape(5, 6, 10, 11, 8), r1.shape);

            var p2 = tf.random.normal(new Shape(4,3));
            var i2 = tf.constant(new int[,] { { 0, 2} });
            var r2 = tf.gather(p2, i2, axis: 0);
            Assert.AreEqual(new Shape(1, 2, 3), r2.shape);

            var r3 = tf.gather(p2, i2, axis: 1);
            Assert.AreEqual(new Shape(4,1,2), r3.shape);
        }

        /// <summary>
        /// https://www.tensorflow.org/api_docs/python/tf/TensorArray
        /// </summary>
        [TestMethod]
        public void TensorArray()
        {
            var ta = tf.TensorArray(tf.float32, size: 0, dynamic_size: true, clear_after_read: false);
            ta.write(0, 10);
            ta.write(1, 20);
            ta.write(2, 30);
            Assert.AreEqual(ta.read(0).numpy(), 10f);
            Assert.AreEqual(ta.read(1).numpy(), 20f);
            Assert.AreEqual(ta.read(2).numpy(), 30f);
        }

        /// <summary>
        /// https://www.tensorflow.org/api_docs/python/tf/reverse
        /// </summary>
        [TestMethod]
        public void ReverseArray()
        {
            var a = tf.random.normal((2, 3));
            var b = tf.reverse(a, -1);
            Assert.IsTrue(Equal(a[0].ToArray<float>().Reverse().ToArray(), b[0].ToArray<float>()));
            Assert.IsTrue(Equal(a[1].ToArray<float>().Reverse().ToArray(), b[1].ToArray<float>()));
        }

        [TestMethod]
        public void ReverseImgArray3D()
        {
            // 创建 sourceImg 数组
            var sourceImgArray = new float[,,] {
            {
                { 237, 28, 36 },
                { 255, 255, 255 },
                { 255, 255, 255 }
            },
            {
                { 255, 255, 255 },
                { 255, 255, 255 },
                { 255, 255, 255 }
            }
        };
            var sourceImg = ops.convert_to_tensor(sourceImgArray);

            // 创建 lrImg 数组
            var lrImgArray = new float[,,] {
            {
                { 255, 255, 255 },
                { 255, 255, 255 },
                { 237, 28, 36 }
            },
            {
                { 255, 255, 255 },
                { 255, 255, 255 },
                { 255, 255, 255 }
            }
        };
            var lrImg = ops.convert_to_tensor(lrImgArray);

            var lr = tf.image.flip_left_right(sourceImg);
            Assert.IsTrue(Equal(lrImg.numpy().ToArray<float>(), lr.numpy().ToArray<float>()), "tf.image.flip_left_right fail.");

            var lr2 = tf.reverse(sourceImg, 1);
            Assert.IsTrue(Equal(lrImg.numpy().ToArray<float>(), lr2.numpy().ToArray<float>()), "tf.reverse (axis=1) fail.");

            var lr3 = gen_array_ops.reverse_v2(sourceImg, ops.convert_to_tensor(new[] { 1 }));
            Assert.IsTrue(Equal(lrImg.numpy().ToArray<float>(), lr3.numpy().ToArray<float>()), "gen_array_ops.reverse_v2 axis=1 fail.");

            // 创建 udImg  数组
            var udImgArray = new float[,,] {
            {
                { 255, 255, 255 },
                { 255, 255, 255 },
                { 255, 255, 255 }
            },
            {
                { 237, 28, 36 },
                { 255, 255, 255 },
                { 255, 255, 255 }
            }
        };
            var udImg = ops.convert_to_tensor(udImgArray);

            var ud = tf.image.flip_up_down(sourceImg);
            Assert.IsTrue(Equal(udImg.numpy().ToArray<float>(), ud.numpy().ToArray<float>()), "tf.image.flip_up_down fail.");

            var ud2 = tf.reverse(sourceImg, new Axis(0));
            Assert.IsTrue(Equal(udImg.numpy().ToArray<float>(), ud2.numpy().ToArray<float>()), "tf.reverse (axis=0) fail.");

            var ud3 = gen_array_ops.reverse_v2(sourceImg, ops.convert_to_tensor(new[] { 0 }));
            Assert.IsTrue(Equal(udImg.numpy().ToArray<float>(), ud3.numpy().ToArray<float>()), "gen_array_ops.reverse_v2 axis=0 fail.");
        }

        [TestMethod]
        public void ReverseImgArray4D()
        {
            // 原图左上角，加一张左右翻转后的图片
            var m = new float[,,,] {
            {
                {
                    { 237, 28, 36 },
                    { 255, 255, 255 },
                    { 255, 255, 255 }
                },
                {
                    { 255, 255, 255 },
                    { 255, 255, 255 },
                    { 255, 255, 255 }
                }
            },
            {
                {
                    { 255, 255, 255 },
                    { 255, 255, 255 },
                    { 237, 28, 36 }
                },
                {
                    { 255, 255, 255 },
                    { 255, 255, 255 },
                    { 255, 255, 255 }
                }
            }
        };
            var sourceImg = ops.convert_to_tensor(m);

            var lrArray = new float[,,,] {
            {
                {
                    { 255, 255, 255 },
                    { 255, 255, 255 },
                    { 237, 28, 36 },
                },
                {
                    { 255, 255, 255 },
                    { 255, 255, 255 },
                    { 255, 255, 255 }
                }
            },
            {
                {
                    { 237, 28, 36 },
                    { 255, 255, 255 },
                    { 255, 255, 255 },
                },
                {
                    { 255, 255, 255 },
                    { 255, 255, 255 },
                    { 255, 255, 255 }
                }
            }
        };
            var lrImg = ops.convert_to_tensor(lrArray);

            // 创建 ud 数组
            var udArray = new float[,,,] {
            {
                {
                    { 255, 255, 255 },
                    { 255, 255, 255 },
                    { 255, 255, 255 }
                },
                {
                    { 237, 28, 36 },
                    { 255, 255, 255 },
                    { 255, 255, 255 }
                }
            },
            {
                {
                    { 255, 255, 255 },
                    { 255, 255, 255 },
                    { 255, 255, 255 }
                },
                {
                    { 255, 255, 255 },
                    { 255, 255, 255 },
                    { 237, 28, 36 }
                }
            }
        };
            var udImg = ops.convert_to_tensor(udArray);

            var ud3 = gen_array_ops.reverse_v2(sourceImg, ops.convert_to_tensor(new[] { 1 }));
            Assert.IsTrue(Equal(udImg.numpy().ToArray<float>(), ud3.numpy().ToArray<float>()), "gen_array_ops.reverse_v2 axis=1 fail.");

            var ud2 = tf.reverse(sourceImg, new Axis(1));
            Assert.IsTrue(Equal(udImg.numpy().ToArray<float>(), ud2.numpy().ToArray<float>()), "tf.reverse (axis=1) fail.");

            var ud = tf.image.flip_up_down(sourceImg);
            Assert.IsTrue(Equal(udImg.numpy().ToArray<float>(), ud.numpy().ToArray<float>()), "tf.image.flip_up_down fail.");

            // 左右翻转
            var lr = tf.image.flip_left_right(sourceImg);
            Assert.IsTrue(Equal(lrImg.numpy().ToArray<float>(), lr.numpy().ToArray<float>()), "tf.image.flip_left_right fail.");

            var lr2 = tf.reverse(sourceImg, 0);
            Assert.IsTrue(Equal(lrImg.numpy().ToArray<float>(), lr2.numpy().ToArray<float>()), "tf.reverse (axis=1) fail.");

            var lr3 = gen_array_ops.reverse_v2(sourceImg, ops.convert_to_tensor(new[] { 0 }));
            Assert.IsTrue(Equal(lrImg.numpy().ToArray<float>(), lr3.numpy().ToArray<float>()), "gen_array_ops.reverse_v2 axis=1 fail.");

        }

        [TestMethod]
        public void ReverseImgArray4D_3x3()
        {
            // 原图左上角，加一张左右翻转后的图片
            var m = new float[,,,] {
            {
                {
                    { 237, 28, 36 },
                    { 255, 255, 255 },
                    { 255, 255, 255 }
                },
                {
                    { 255, 255, 255 },
                    { 255, 255, 255 },
                    { 255, 255, 255 }
                },
                {
                    { 255, 255, 255 },
                    { 255, 255, 255 },
                    { 255, 255, 255 }
                }
            },
            {
                {
                    { 255, 255, 255 },
                    { 255, 255, 255 },
                    { 237, 28, 36 }
                },
                {
                    { 255, 255, 255 },
                    { 255, 255, 255 },
                    { 255, 255, 255 }
                },
                {
                    { 255, 255, 255 },
                    { 255, 255, 255 },
                    { 255, 255, 255 }
                }
            }
        };
            var sourceImg = ops.convert_to_tensor(m);

            var lrArray = new float[,,,] {
            {
                {
                    { 255, 255, 255 },
                    { 255, 255, 255 },
                    { 237, 28, 36 },
                },
                {
                    { 255, 255, 255 },
                    { 255, 255, 255 },
                    { 255, 255, 255 }
                },
                {
                    { 255, 255, 255 },
                    { 255, 255, 255 },
                    { 255, 255, 255 }
                }
            },
            {
                {
                    { 237, 28, 36 },
                    { 255, 255, 255 },
                    { 255, 255, 255 },
                },
                {
                    { 255, 255, 255 },
                    { 255, 255, 255 },
                    { 255, 255, 255 }
                },
                {
                    { 255, 255, 255 },
                    { 255, 255, 255 },
                    { 255, 255, 255 }
                }
            }
        };
            var lrImg = ops.convert_to_tensor(lrArray);

            // 创建 ud 数组
            var udArray = new float[,,,] {
            {
                {
                    { 255, 255, 255 },
                    { 255, 255, 255 },
                    { 255, 255, 255 }
                },
                                {
                    { 255, 255, 255 },
                    { 255, 255, 255 },
                    { 255, 255, 255 }
                },
                {
                    { 237, 28, 36 },
                    { 255, 255, 255 },
                    { 255, 255, 255 }
                }
            },
            {                {
                    { 255, 255, 255 },
                    { 255, 255, 255 },
                    { 255, 255, 255 }
                },
                {
                    { 255, 255, 255 },
                    { 255, 255, 255 },
                    { 255, 255, 255 }
                },
                {
                    { 255, 255, 255 },
                    { 255, 255, 255 },
                    { 237, 28, 36 }
                }
            }
        };
            var udImg = ops.convert_to_tensor(udArray);

            var ud3 = gen_array_ops.reverse_v2(sourceImg, ops.convert_to_tensor(new[] { 1 }));
            Assert.IsTrue(Equal(udImg.numpy().ToArray<float>(), ud3.numpy().ToArray<float>()), "gen_array_ops.reverse_v2 axis=1 fail.");

            var ud2 = tf.reverse(sourceImg, new Axis(1));
            Assert.IsTrue(Equal(udImg.numpy().ToArray<float>(), ud2.numpy().ToArray<float>()), "tf.reverse (axis=1) fail.");

            var ud = tf.image.flip_up_down(sourceImg);
            Assert.IsTrue(Equal(udImg.numpy().ToArray<float>(), ud.numpy().ToArray<float>()), "tf.image.flip_up_down fail.");

            // 左右翻转
            var lr = tf.image.flip_left_right(sourceImg);
            Assert.IsTrue(Equal(lrImg.numpy().ToArray<float>(), lr.numpy().ToArray<float>()), "tf.image.flip_left_right fail.");

            var lr2 = tf.reverse(sourceImg, 0);
            Assert.IsTrue(Equal(lrImg.numpy().ToArray<float>(), lr2.numpy().ToArray<float>()), "tf.reverse (axis=1) fail.");

            var lr3 = gen_array_ops.reverse_v2(sourceImg, ops.convert_to_tensor(new[] { 0 }));
            Assert.IsTrue(Equal(lrImg.numpy().ToArray<float>(), lr3.numpy().ToArray<float>()), "gen_array_ops.reverse_v2 axis=1 fail.");

        }
    }
}
