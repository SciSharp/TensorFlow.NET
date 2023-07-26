using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow.NumPy;
using Tensorflow;
using static Tensorflow.Binding;
using System.Linq;

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
            var input_array = tf.constant(np.array(new int[] { 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6 }).reshape((3, 2, 3)));
            var indices = tf.constant(np.array(new int[] { 0, 2 }));

            var r1 = array_ops.slice(input_array, ops.convert_n_to_tensor(new object[] { 1, 0, 0 }), ops.convert_n_to_tensor(new object[] { 1, 1, 3 }));
            Assert.AreEqual(new Shape(1, 1, 3), r1.shape);
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
            var r1 = tf.gather(p1, i1, axis: 2);
            Assert.AreEqual(new Shape(5, 6, 10, 11, 8), r1.shape);

            var p2 = tf.random.normal(new Shape(4, 3));
            var i2 = tf.constant(new int[,] { { 0, 2 } });
            var r2 = tf.gather(p2, i2, axis: 0);
            Assert.AreEqual(new Shape(1, 2, 3), r2.shape);

            var r3 = tf.gather(p2, i2, axis: 1);
            Assert.AreEqual(new Shape(4, 1, 2), r3.shape);
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
        /// 
        /// </summary>
        [TestMethod]
        public void Reverse()
        {
            /*
             * python run get test data code:
             
import tensorflow as tf

data=[[1, 2, 3], [4, 5, 6], [7,8,9]]

data2 = tf.constant(data)

print('test data shaper:', data2.shape)
print('test data:', data2)

axis = [-2,-1,0,1]
for i in axis:
    print('')
    print('axis:', i)
    ax = tf.constant([i])
    datar = tf.reverse(data2, ax)
    datar2 = array_ops.reverse(data2, ax)
    print(datar)
    print(datar2)

             * */
            var inputData = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
            var expectedOutput = new[] {
            // np.array(new int[,] { { 7, 8, 9 }, { 4, 5, 6 }, { 1, 2, 3 } }),
            np.array(new int[,] { { 3, 2, 1 }, { 6, 5, 4 }, { 9, 8, 7 } }),
            np.array(new int[,] { { 7, 8, 9 }, { 4, 5, 6 }, { 1, 2, 3 } }),
            np.array(new int[,] { { 3, 2, 1 }, { 6, 5, 4 }, { 9, 8, 7 } })
        };

            var axes = new int [] {  
                -1, 
                0,
                1 };
            for (var i = 0; i < axes.Length; i++)
            {
                var axis = axes[i];
                var expected = tf.constant(expectedOutput[i]).numpy();

                var inputTensor = tf.constant(inputData);
                var axisTrensor = tf.constant(new[] { axis });

                var outputTensor = tf.reverse_v2(inputTensor, axisTrensor);
                var npout = outputTensor.numpy();
                Assert.IsTrue(Enumerable.SequenceEqual(npout, expected), $"axis:{axis}");

                var outputTensor2 = tf.reverse_v2(inputTensor, new[] { axis } );
                var npout2 = outputTensor2.numpy();
                Assert.IsTrue(Enumerable.SequenceEqual(npout2, expected), $"axis:{axis}");

            }
        }
    }
}
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow.NumPy;
using Tensorflow;
using static Tensorflow.Binding;
using System.Linq;

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
    }
}
