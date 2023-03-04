using Microsoft.VisualStudio.TestPlatform.Utilities;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;
using Tensorflow.Operations;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow.NumPy;
using Microsoft.VisualBasic;
using static HDF.PInvoke.H5T;
using Tensorflow.Keras.UnitTest.Helpers;
using Tensorflow.Keras.Optimizers;

namespace Tensorflow.Keras.UnitTest
{
    [TestClass]
    public class MultiInputModelTest
    {
        [TestMethod]
        public void SimpleModel()
        {
            var inputs = keras.Input((28, 28, 1));
            var conv1 = keras.layers.Conv2D(16, (3, 3), activation: "relu", padding: "same").Apply(inputs);
            var pool1 = keras.layers.MaxPooling2D((2, 2), 2).Apply(conv1);
            var conv2 = keras.layers.Conv2D(32, (3, 3), activation: "relu", padding: "same").Apply(pool1);
            var pool2 = keras.layers.MaxPooling2D((2, 2), 2).Apply(conv2);
            var flat1 = keras.layers.Flatten().Apply(pool2);

            var inputs_2 = keras.Input((28, 28, 1));
            var conv1_2 = keras.layers.Conv2D(16, (3, 3), activation: "relu", padding: "same").Apply(inputs_2);
            var pool1_2 = keras.layers.MaxPooling2D((4, 4), 4).Apply(conv1_2);
            var conv2_2 = keras.layers.Conv2D(32, (1, 1), activation: "relu", padding: "same").Apply(pool1_2);
            var pool2_2 = keras.layers.MaxPooling2D((2, 2), 2).Apply(conv2_2);
            var flat1_2 = keras.layers.Flatten().Apply(pool2_2);

            var concat = keras.layers.Concatenate().Apply((flat1, flat1_2));
            var dense1 = keras.layers.Dense(512, activation: "relu").Apply(concat);
            var dense2 = keras.layers.Dense(128, activation: "relu").Apply(dense1);
            var dense3 = keras.layers.Dense(10, activation:  "relu").Apply(dense2);
            var output = keras.layers.Softmax(-1).Apply(dense3);

            var model = keras.Model((inputs, inputs_2), output);
            model.summary();

            var data_loader = new MnistModelLoader();

            var dataset = data_loader.LoadAsync(new ModelLoadSetting
            {
                TrainDir = "mnist",
                OneHot = false,
                ValidationSize = 59000,
            }).Result;

            var loss = keras.losses.SparseCategoricalCrossentropy();
            var optimizer = new Adam(0.001f);
            model.compile(optimizer, loss, new string[] { "accuracy" });

            NDArray x1 = np.reshape(dataset.Train.Data, (dataset.Train.Data.shape[0], 28, 28, 1));
            NDArray x2 = x1;

            var x = new NDArray[] { x1, x2 };
            model.fit(x, dataset.Train.Labels, batch_size: 8, epochs: 3);
        }
    }
}
