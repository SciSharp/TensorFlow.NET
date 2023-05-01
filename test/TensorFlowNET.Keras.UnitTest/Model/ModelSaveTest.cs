using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;
using System.Diagnostics;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Models;
using Tensorflow.Keras.Optimizers;
using Tensorflow.Keras.Saving;
using Tensorflow.Keras.UnitTest.Helpers;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.UnitTest.Model
{
    /// <summary>
    /// https://www.tensorflow.org/guide/keras/save_and_serialize
    /// </summary>
    [TestClass]
    public class ModelSaveTest : EagerModeTestBase
    {
        [TestMethod]
        public void GetAndFromConfig()
        {
            var model = GetFunctionalModel();
            var config = model.get_config();
            Debug.Assert(config is FunctionalConfig);
            var new_model = new ModelsApi().from_config(config as FunctionalConfig);
            Assert.AreEqual(model.Layers.Count, new_model.Layers.Count);
        }

        IModel GetFunctionalModel()
        {
            // Create a simple model.
            var inputs = keras.Input(shape: 32);
            var dense_layer = keras.layers.Dense(1);
            var outputs = dense_layer.Apply(inputs);
            return keras.Model(inputs, outputs);
        }

        [TestMethod]
        public void SimpleModelFromAutoCompile()
        {
            var inputs = tf.keras.layers.Input((28, 28, 1));
            var x = tf.keras.layers.Flatten().Apply(inputs);
            x = tf.keras.layers.Dense(100, activation: "relu").Apply(x);
            x = tf.keras.layers.Dense(units: 10).Apply(x);
            var outputs = tf.keras.layers.Softmax(axis: 1).Apply(x);
            var model = tf.keras.Model(inputs, outputs);

            model.compile(new Adam(0.001f),
                tf.keras.losses.SparseCategoricalCrossentropy(),
                new string[] { "accuracy" });

            var data_loader = new MnistModelLoader();
            var num_epochs = 1;
            var batch_size = 50;

            var dataset = data_loader.LoadAsync(new ModelLoadSetting
            {
                TrainDir = "mnist",
                OneHot = false,
                ValidationSize = 58000,
            }).Result;

            model.fit(dataset.Train.Data, dataset.Train.Labels, batch_size, num_epochs);

            model.save("./pb_simple_compile", save_format: "tf");
        }

        [TestMethod]
        public void SimpleModelFromSequential()
        {
            var model = keras.Sequential(new List<ILayer>()
            {
                tf.keras.layers.InputLayer((28, 28, 1)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(100, "relu"),
                tf.keras.layers.Dense(10),
                tf.keras.layers.Softmax()
            });

            model.summary();

            model.compile(new Adam(0.001f), tf.keras.losses.SparseCategoricalCrossentropy(), new string[] { "accuracy" });

            var data_loader = new MnistModelLoader();
            var num_epochs = 1;
            var batch_size = 50;

            var dataset = data_loader.LoadAsync(new ModelLoadSetting
            {
                TrainDir = "mnist",
                OneHot = false,
                ValidationSize = 58000,
            }).Result;

            model.fit(dataset.Train.Data, dataset.Train.Labels, batch_size, num_epochs);

            model.save("./pb_simple_sequential", save_format: "tf");
        }

        [TestMethod]
        public void AlexnetFromSequential()
        {
            var model = keras.Sequential(new List<ILayer>()
            {
                tf.keras.layers.InputLayer((227, 227, 3)),
                tf.keras.layers.Conv2D(96, (11, 11), (4, 4), activation:"relu", padding:"valid"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((3, 3), strides:(2, 2)),

                tf.keras.layers.Conv2D(256, (5, 5), (1, 1), "same", activation: "relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((3, 3), (2, 2)),

                tf.keras.layers.Conv2D(384, (3, 3), (1, 1), "same", activation: "relu"),
                tf.keras.layers.BatchNormalization(),

                tf.keras.layers.Conv2D(384, (3, 3), (1, 1), "same", activation: "relu"),
                tf.keras.layers.BatchNormalization(),

                tf.keras.layers.Conv2D(256, (3, 3), (1, 1), "same", activation: "relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((3, 3), (2, 2)),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4096, activation: "relu"),
                tf.keras.layers.Dropout(0.5f),

                tf.keras.layers.Dense(4096, activation: "relu"),
                tf.keras.layers.Dropout(0.5f),

                tf.keras.layers.Dense(1000, activation: "linear"),
                tf.keras.layers.Softmax(1)
            });

            model.compile(tf.keras.optimizers.Adam(), tf.keras.losses.SparseCategoricalCrossentropy(from_logits: true), new string[] { "accuracy" });

            var num_epochs = 1;
            var batch_size = 8;

            var dataset = new RandomDataSet(new Shape(227, 227, 3), 16);

            model.fit(dataset.Data, dataset.Labels, batch_size, num_epochs);

            model.save("./alexnet_from_sequential", save_format: "tf");

            // The saved model can be test with the following python code:
            #region alexnet_python_code
            //import pathlib
            //import tensorflow as tf

            //def func(a):
            //    return -a

            //if __name__ == '__main__':
            //    model = tf.keras.models.load_model("./pb_alex_sequential")
            //    model.summary()

            //    num_classes = 5
            //    batch_size = 128
            //    img_height = 227
            //    img_width = 227
            //    epochs = 100

            //    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
            //    data_dir = tf.keras.utils.get_file('flower_photos', origin = dataset_url, untar = True)
            //    data_dir = pathlib.Path(data_dir)

            //    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            //        data_dir,
            //        validation_split = 0.2,
            //        subset = "training",
            //        seed = 123,
            //        image_size = (img_height, img_width),
            //        batch_size = batch_size)

            //    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            //        data_dir,
            //        validation_split = 0.2,
            //        subset = "validation",
            //        seed = 123,
            //        image_size = (img_height, img_width),
            //        batch_size = batch_size)


            //    model.compile(optimizer = 'adam',
            //                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
            //                  metrics =['accuracy'])

            //    model.build((None, img_height, img_width, 3))

            //    history = model.fit(
            //        train_ds,
            //        validation_data = val_ds,
            //        epochs = epochs
            //    )
            #endregion
        }

        [TestMethod]
        public void SaveAfterLoad()
        {
            var model = tf.keras.models.load_model(@"Assets/simple_model_from_auto_compile");
            model.summary();

            model.save("Assets/saved_auto_compile_after_loading");

            //model = tf.keras.models.load_model(@"Assets/saved_auto_compile_after_loading");
            //model.summary();
        }
    }
}
