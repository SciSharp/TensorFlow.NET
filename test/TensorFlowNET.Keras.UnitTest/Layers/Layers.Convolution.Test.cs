using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using Tensorflow.NumPy;
using static Tensorflow.KerasApi;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.UnitTest.Layers
{
    [TestClass]
    public class LayersConvolutionTest : EagerModeTestBase
    {
        [TestMethod]
        public void BasicConv1D()
        {
            var filters = 8;

            var conv = keras.layers.Conv1D(filters, kernel_size: 3, activation: "linear");

            var x = np.arange(256.0f).reshape((8, 8, 4));
            var y = conv.Apply(x);

            Assert.AreEqual(y.shape, (8, 6, 8));
            Assert.AreEqual(filters, y.shape[2]);
        }

        [TestMethod]
        public void BasicConv1D_ksize()
        {
            var filters = 8;

            var conv = keras.layers.Conv1D(filters, kernel_size: 3, activation: "linear");

            var x = np.arange(256.0f).reshape((8, 8, 4));
            var y = conv.Apply(x);

            Assert.AreEqual(3, y.shape.ndim);
            Assert.AreEqual(x.dims[0], y.shape[0]);
            Assert.AreEqual(x.dims[1] - 2, y.shape[1]);
            Assert.AreEqual(filters, y.shape[2]);
        }

        [TestMethod]
        public void BasicConv1D_ksize_same()
        {
            var filters = 8;

            var conv = keras.layers.Conv1D(filters, kernel_size: 3, padding: "same", activation: "linear");

            var x = np.arange(256.0f).reshape((8, 8, 4));
            var y = conv.Apply(x);

            Assert.AreEqual(3, y.shape.ndim);
            Assert.AreEqual(x.dims[0], y.shape[0]);
            Assert.AreEqual(x.dims[1], y.shape[1]);
            Assert.AreEqual(filters, y.shape[2]);
        }

        [TestMethod]
        public void BasicConv1D_ksize_strides()
        {
            var filters = 8;
            var conv = keras.layers.Conv1D(filters, kernel_size: 3, strides: 2, activation: "linear");

            var x = np.arange(256.0f).reshape((8, 8, 4));
            var y = conv.Apply(x);

            Assert.AreEqual(3, y.shape.ndim);
            Assert.AreEqual(x.dims[0], y.shape[0]);
            Assert.AreEqual(x.dims[1] - 5, y.shape[1]);
            Assert.AreEqual(filters, y.shape[2]);
        }

        [TestMethod]
        public void BasicConv1D_ksize_dilations()
        {
            var filters = 8;
            var conv = keras.layers.Conv1D(filters, kernel_size: 3, dilation_rate: 2, activation: "linear");

            var x = np.arange(256.0f).reshape((8, 8, 4));
            var y = conv.Apply(x);

            Assert.AreEqual(3, y.shape.ndim);
            Assert.AreEqual(x.dims[0], y.shape[0]);
            Assert.AreEqual(x.dims[1] - 4, y.shape[1]);
            Assert.AreEqual(filters, y.shape[2]);
        }

        [TestMethod]
        public void BasicConv1D_ksize_dilation_same()
        {
            var filters = 8;
            var conv = keras.layers.Conv1D(filters, kernel_size: 3, dilation_rate: 2, padding: "same", activation: "linear");

            var x = np.arange(256.0f).reshape((8, 8, 4));
            var y = conv.Apply(x);

            Assert.AreEqual(3, y.shape.ndim);
            Assert.AreEqual(x.dims[0], y.shape[0]);
            Assert.AreEqual(x.dims[1], y.shape[1]);
            Assert.AreEqual(filters, y.shape[2]);
        }

        [TestMethod]
        public void BasicConv2D()
        {
            var filters = 8;
            var conv = keras.layers.Conv2D(filters, activation: "linear");

            var x = np.arange(256.0f).reshape((1, 8, 8, 4));
            var y = conv.Apply(x);

            Assert.AreEqual(4, y.shape.ndim);
            Assert.AreEqual(x.dims[0], y.shape[0]);
            Assert.AreEqual(x.dims[1] - 4, y.shape[1]);
            Assert.AreEqual(x.dims[2] - 4, y.shape[2]);
            Assert.AreEqual(filters, y.shape[3]);
        }

        [TestMethod]
        public void BasicConv2D_ksize()
        {
            var filters = 8;
            var conv = keras.layers.Conv2D(filters, kernel_size: 3, activation: "linear");

            var x = np.arange(256.0f).reshape((1, 8, 8, 4));
            var y = conv.Apply(x);

            Assert.AreEqual(4, y.shape.ndim);
            Assert.AreEqual(x.dims[0], y.shape[0]);
            Assert.AreEqual(x.dims[1] - 2, y.shape[1]);
            Assert.AreEqual(x.dims[2] - 2, y.shape[2]);
            Assert.AreEqual(filters, y.shape[3]);
        }

        [TestMethod]
        public void BasicConv2D_ksize_same()
        {
            var filters = 8;
            var conv = keras.layers.Conv2D(filters, kernel_size: 3, padding: "same", activation: "linear");

            var x = np.arange(256.0f).reshape((1, 8, 8, 4));
            var y = conv.Apply(x);

            Assert.AreEqual(4, y.shape.ndim);
            Assert.AreEqual(x.dims[0], y.shape[0]);
            Assert.AreEqual(x.dims[1], y.shape[1]);
            Assert.AreEqual(x.dims[2], y.shape[2]);
            Assert.AreEqual(filters, y.shape[3]);
        }

        [TestMethod]
        public void BasicConv2D_ksize_strides()
        {
            var filters = 8;
            var conv = keras.layers.Conv2D(filters, kernel_size: 3, strides: 2, activation: "linear");

            var x = np.arange(256.0f).reshape((1, 8, 8, 4));
            var y = conv.Apply(x);

            Assert.AreEqual(4, y.shape.ndim);
            Assert.AreEqual(x.dims[0], y.shape[0]);
            Assert.AreEqual(x.dims[1] - 5, y.shape[1]);
            Assert.AreEqual(x.dims[2] - 5, y.shape[2]);
            Assert.AreEqual(filters, y.shape[3]);
        }

        [TestMethod]
        public void BasicConv2D_ksize_dilation()
        {
            var filters = 8;
            var conv = keras.layers.Conv2D(filters, kernel_size: 3, dilation_rate: 2, activation: "linear");

            var x = np.arange(256.0f).reshape((1, 8, 8, 4));
            var y = conv.Apply(x);

            Assert.AreEqual(4, y.shape.ndim);
            Assert.AreEqual(x.dims[0], y.shape[0]);
            Assert.AreEqual(x.dims[1] - 4, y.shape[1]);
            Assert.AreEqual(x.dims[2] - 4, y.shape[2]);
            Assert.AreEqual(filters, y.shape[3]);
        }

        [TestMethod]
        public void BasicConv2D_ksize_dilation_same()
        {
            var filters = 8;
            var conv = keras.layers.Conv2D(filters, kernel_size: 3, dilation_rate: 2, padding: "same", activation: "linear");

            var x = np.arange(256.0f).reshape((1, 8, 8, 4));
            var y = conv.Apply(x);

            Assert.AreEqual(4, y.shape.ndim);
            Assert.AreEqual(x.dims[0], y.shape[0]);
            Assert.AreEqual(x.dims[1], y.shape[1]);
            Assert.AreEqual(x.dims[2], y.shape[2]);
            Assert.AreEqual(filters, y.shape[3]);
        }


        [TestMethod]
        public void BasicDepthwiseConv2D()
        {
            var conv = keras.layers.DepthwiseConv2D(kernel_size:3, strides:1, activation: null, 
                padding:"same", depthwise_initializer: "ones");

            var x = np.arange(2 * 9* 9* 3).reshape((2, 9, 9, 3));
            var x2 = ops.convert_to_tensor(x, TF_DataType.TF_FLOAT);

            var y = conv.Apply(x2);

            print($"input:{x2.shape} DepthwiseConv2D.out: {y.shape}");


            Assert.AreEqual(4, y.shape.ndim);
            var arr = y.numpy().reshape((2, 9, 9, 3));

            AssertArray(x[new int[] { 1, 1, 1 }].ToArray<int>(), new int[] { 273, 274, 275 });
            AssertArray(arr[new int[] { 1, 1, 1 }].ToArray<float>(), new float[] { 2457f, 2466f, 2475f });

            var bn = keras.layers.BatchNormalization();
            var y2 = bn.Apply(y);
            arr = y2.numpy().ToArray<float>();

            double delta = 0.0001; // 误差范围

            Assert.AreEqual(arr[0], 59.97002f, delta);
            Assert.AreEqual(arr[1], 63.96802f, delta);
        }


        [TestMethod]
        public void BasicDepthwiseConv2D_strides_2()
        {
            var conv = keras.layers.DepthwiseConv2D(kernel_size: 3, strides: (1, 2, 2, 1), activation: null,
                padding: "same", depthwise_initializer: "ones");

            var x = np.arange(2 * 9 * 9 * 3).reshape((2, 9, 9, 3));
            var x2 = ops.convert_to_tensor(x, TF_DataType.TF_FLOAT);

            var y = conv.Apply(x2);

            print($"input:{x2.shape} DepthwiseConv2D.out: {y.shape}");

            Assert.AreEqual(4, y.shape.ndim);
            var arr = y.numpy().reshape((2, 5, 5, 3));

            AssertArray(x[new int[] { 1, 1, 1 }].ToArray<int>(), new int[] { 273, 274, 275 });
            AssertArray(arr[new int[] { 1, 1, 1 }].ToArray<float>(), new float[] { 2727f, 2736f, 2745f });

            var bn = keras.layers.BatchNormalization();
            var y2 = bn.Apply(y);
            arr = y2.numpy().ToArray<float>();

            double delta = 0.0001; // 误差范围

            Assert.AreEqual(arr[0], 59.97002f, delta);
            Assert.AreEqual(arr[1], 63.96802f, delta);
        }



        [TestMethod]
        public void BasicDepthwiseConv2D_strides_3()
        {
            var conv = keras.layers.DepthwiseConv2D(kernel_size: 3, strides: 3, activation: null,
                padding: "same", depthwise_initializer: "ones");

            var x = np.arange(2 * 9 * 9 * 3).reshape((2, 9, 9, 3));
            var x2 = ops.convert_to_tensor(x, TF_DataType.TF_FLOAT);

            var y = conv.Apply(x2);

            print($"input:{x2.shape} DepthwiseConv2D.out: {y.shape}");

            Assert.AreEqual(4, y.shape.ndim);
            var arr = y.numpy().reshape((2, 3, 3, 3));

            AssertArray(x[new int[] { 1, 1, 1 }].ToArray<int>(), new int[] { 273, 274, 275 });
            AssertArray(arr[new int[] { 1, 1, 1 }].ToArray<float>(), new float[] { 3267f, 3276f, 3285f });

            var bn = keras.layers.BatchNormalization();
            var y2 = bn.Apply(y);
            arr = y2.numpy().ToArray<float>();

            double delta = 0.0001; // 误差范围
              
            Assert.AreEqual(arr[0], 269.86508f, delta);
            Assert.AreEqual(arr[1], 278.8606f, delta);

        }
        [TestMethod]
        public void BasicDepthwiseConv2D_UseBias()
        {
            var conv = keras.layers.DepthwiseConv2D(kernel_size: 3, strides: 1, activation: null,
                use_bias: true, padding: "same",
                depthwise_initializer: "ones",
                bias_initializer:"ones"
                );

            var weight = conv.get_weights();

            var x = np.arange(9 * 9 * 3).reshape((1, 9, 9, 3));
            var x2 = ops.convert_to_tensor(x, TF_DataType.TF_FLOAT);
            var y = conv.Apply(x2);

            Assert.AreEqual(4, y.shape.ndim);
            var arr = y.numpy().ToArray<float>();

            Assert.AreEqual(arr[0], 61f);
            Assert.AreEqual(arr[1], 65f);

            var bn = keras.layers.BatchNormalization();
            var y2 = bn.Apply(y);
            arr = y2.numpy().ToArray<float>();

            double delta = 0.0001; // 误差范围

            Assert.AreEqual(arr[0], 60.96952f, delta);
            Assert.AreEqual(arr[1], 64.96752f, delta);
        }
    }
}
