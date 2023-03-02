using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;
using System.Diagnostics;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Optimizers;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Keras.UnitTest.SaveModel;

[TestClass]
public class SequentialModelSave
{
    [TestMethod]
    public void SimpleModelFromAutoCompile()
    {
        var inputs = new KerasInterface().Input((28, 28, 1));
        var x = new Flatten(new FlattenArgs()).Apply(inputs);
        x = new Dense(new DenseArgs() { Units = 100, Activation = tf.nn.relu }).Apply(x);
        x = new LayersApi().Dense(units: 10).Apply(x);
        var outputs = new LayersApi().Softmax(axis: 1).Apply(x);
        var model = new KerasInterface().Model(inputs, outputs);

        model.compile(new Adam(0.001f), new LossesApi().SparseCategoricalCrossentropy(), new string[] { "accuracy" });

        var data_loader = new MnistModelLoader();
        var num_epochs = 1;
        var batch_size = 50;

        var dataset = data_loader.LoadAsync(new ModelLoadSetting
        {
            TrainDir = "mnist",
            OneHot = false,
            ValidationSize = 10000,
        }).Result;

        model.fit(dataset.Train.Data, dataset.Train.Labels, batch_size, num_epochs);

        model.save("./pb_simple_compile", save_format: "tf");
    }

    [TestMethod]
    public void SimpleModelFromSequential()
    {
        Model model = KerasApi.keras.Sequential(new List<ILayer>()
        {
            keras.layers.InputLayer((28, 28, 1)),
            keras.layers.Flatten(),
            keras.layers.Dense(100, "relu"),
            keras.layers.Dense(10),
            keras.layers.Softmax(1)
        });

        model.summary();

        model.compile(new Adam(0.001f), new LossesApi().SparseCategoricalCrossentropy(), new string[] { "accuracy" });

        var data_loader = new MnistModelLoader();
        var num_epochs = 1;
        var batch_size = 50;

        var dataset = data_loader.LoadAsync(new ModelLoadSetting
        {
            TrainDir = "mnist",
            OneHot = false,
            ValidationSize = 50000,
        }).Result;

        model.fit(dataset.Train.Data, dataset.Train.Labels, batch_size, num_epochs);

        model.save("./pb_simple_sequential", save_format: "tf");
    }

    [TestMethod]
    public void AlexnetFromSequential()
    {
        Model model = KerasApi.keras.Sequential(new List<ILayer>()
        {
            keras.layers.InputLayer((227, 227, 3)),
            keras.layers.Conv2D(96, (11, 11), (4, 4), activation:"relu", padding:"valid"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((3, 3), strides:(2, 2)),

            keras.layers.Conv2D(256, (5, 5), (1, 1), "same", activation: "relu"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((3, 3), (2, 2)),

            keras.layers.Conv2D(384, (3, 3), (1, 1), "same", activation: "relu"),
            keras.layers.BatchNormalization(),

            keras.layers.Conv2D(384, (3, 3), (1, 1), "same", activation: "relu"),
            keras.layers.BatchNormalization(),

            keras.layers.Conv2D(256, (3, 3), (1, 1), "same", activation: "relu"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((3, 3), (2, 2)),

            keras.layers.Flatten(),
            keras.layers.Dense(4096, activation: "relu"),
            keras.layers.Dropout(0.5f),

            keras.layers.Dense(4096, activation: "relu"),
            keras.layers.Dropout(0.5f),

            keras.layers.Dense(1000, activation: "linear"),
            keras.layers.Softmax(1)
        });

        model.compile(new Adam(0.001f), new LossesApi().SparseCategoricalCrossentropy(from_logits: true), new string[] { "accuracy" });

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

    public class RandomDataSet : DataSetBase
    {
        private Shape _shape;

        public RandomDataSet(Shape shape, int count)
        {
            _shape = shape;
            Debug.Assert(_shape.ndim == 3);
            long[] dims = new long[4];
            dims[0] = count;
            for (int i = 1; i < 4; i++)
            {
                dims[i] = _shape[i - 1];
            }
            Shape s = new Shape(dims);
            Data = np.random.normal(0, 2, s);
            Labels = np.random.uniform(0, 1, (count, 1));
        }
    }
}