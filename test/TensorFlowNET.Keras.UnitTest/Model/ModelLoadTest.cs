using Microsoft.VisualStudio.TestPlatform.Utilities;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Newtonsoft.Json.Linq;
using System.Collections.Generic;
using System.Linq;
using System.Xml.Linq;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Optimizers;
using Tensorflow.Keras.UnitTest.Helpers;
using Tensorflow.NumPy;
using static HDF.PInvoke.H5Z;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.UnitTest.Model;

[TestClass]
public class ModelLoadTest
{
    [TestMethod]
    public void SimpleModelFromAutoCompile()
    {
        var model = tf.keras.models.load_model(@"Assets/simple_model_from_auto_compile");
        model.summary();

        model.compile(new Adam(0.0001f), tf.keras.losses.SparseCategoricalCrossentropy(), new string[] { "accuracy" });

        // check the weights
        var kernel1 = np.load(@"Assets/simple_model_from_auto_compile/kernel1.npy");
        var bias0 = np.load(@"Assets/simple_model_from_auto_compile/bias0.npy");

        Assert.IsTrue(kernel1.Zip(model.TrainableWeights[2].numpy()).All(x => x.First == x.Second));
        Assert.IsTrue(bias0.Zip(model.TrainableWeights[1].numpy()).All(x => x.First == x.Second));

        var data_loader = new MnistModelLoader();
        var num_epochs = 1;
        var batch_size = 8;

        var dataset = data_loader.LoadAsync(new ModelLoadSetting
        {
            TrainDir = "mnist",
            OneHot = false,
            ValidationSize = 58000,
        }).Result;

        model.fit(dataset.Train.Data, dataset.Train.Labels, batch_size, num_epochs);
    }

    [TestMethod]
    public void AlexnetFromSequential()
    {
        new ModelSaveTest().AlexnetFromSequential();
        var model = tf.keras.models.load_model(@"./alexnet_from_sequential");
        model.summary();

        model.compile(new Adam(0.001f), tf.keras.losses.SparseCategoricalCrossentropy(from_logits: true), new string[] { "accuracy" });

        var num_epochs = 1;
        var batch_size = 8;

        var dataset = new RandomDataSet(new Shape(227, 227, 3), 16);

        model.fit(dataset.Data, dataset.Labels, batch_size, num_epochs);
    }

    [TestMethod]
    public void ModelWithSelfDefinedModule()
    {
        var model = tf.keras.models.load_model(@"Assets/python_func_model");
        model.summary();

        model.compile(tf.keras.optimizers.Adam(), tf.keras.losses.SparseCategoricalCrossentropy(), new string[] { "accuracy" });

        var data_loader = new MnistModelLoader();
        var num_epochs = 1;
        var batch_size = 8;

        var dataset = data_loader.LoadAsync(new ModelLoadSetting
        {
            TrainDir = "mnist",
            OneHot = false,
            ValidationSize = 55000,
        }).Result;

        model.fit(dataset.Train.Data, dataset.Train.Labels, batch_size, num_epochs);
    }

    [Ignore]
    [TestMethod]
    public void LSTMLoad()
    {
        var model = tf.keras.models.load_model(@"Assets/lstm_from_sequential");
        model.summary();
        model.compile(tf.keras.optimizers.Adam(), tf.keras.losses.MeanSquaredError(), new string[] { "accuracy" });
        var inputs = tf.random.normal(shape: (10, 5, 3));
        var outputs = tf.random.normal(shape: (10, 1));
        model.fit(inputs.numpy(), outputs.numpy(), batch_size: 10, epochs: 5, workers: 16, use_multiprocessing: true);
    }

    [Ignore]
    [TestMethod]
    public void VGG19()
    {
        var model = tf.keras.models.load_model(@"D:\development\tf.net\models\VGG19");
        model.summary();

        var classify_model = keras.Sequential(new System.Collections.Generic.List<ILayer>()
        {
            model,
            keras.layers.Flatten(),
            keras.layers.Dense(10),
        });
        classify_model.summary();

        classify_model.compile(tf.keras.optimizers.Adam(), tf.keras.losses.SparseCategoricalCrossentropy(), new string[] { "accuracy" });

        var x = np.random.uniform(0, 1, (8, 512, 512, 3));
        var y = np.ones(8);

        classify_model.fit(x, y, batch_size: 4);
    }

    [Ignore]
    [TestMethod]
    public void TestModelBeforeTF2_5()
    {
        var a = keras.layers;
        var model = tf.saved_model.load(@"D:\development\temp\saved_model") as Tensorflow.Keras.Engine.Model;
        model.summary();
    }


    [TestMethod]
    public void BiasRegularizerSaveAndLoad()
    {
        var savemodel = keras.Sequential(new List<ILayer>()
            {
                tf.keras.layers.InputLayer((227, 227, 3)),
                tf.keras.layers.Conv2D(96, (11, 11), (4, 4), activation:"relu", padding:"valid"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((3, 3), strides:(2, 2)),

                tf.keras.layers.Conv2D(256, (5, 5), (1, 1), "same", activation: keras.activations.Relu, bias_regularizer:keras.regularizers.L1L2),
                tf.keras.layers.BatchNormalization(),

                tf.keras.layers.Conv2D(256, (5, 5), (1, 1), "same", activation: keras.activations.Relu, bias_regularizer:keras.regularizers.L2),
                tf.keras.layers.BatchNormalization(),

                tf.keras.layers.Conv2D(256, (5, 5), (1, 1), "same", activation: keras.activations.Relu, bias_regularizer:keras.regularizers.L1),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((3, 3), (2, 2)),

                tf.keras.layers.Flatten(),

                tf.keras.layers.Dense(1000, activation: "linear"),
                tf.keras.layers.Softmax(1)
            });

        savemodel.compile(tf.keras.optimizers.Adam(), tf.keras.losses.SparseCategoricalCrossentropy(from_logits: true), new string[] { "accuracy" });

        var num_epochs = 1;
        var batch_size = 8;

        var trainDataset = new RandomDataSet(new Shape(227, 227, 3), 16);

        savemodel.fit(trainDataset.Data, trainDataset.Labels, batch_size, num_epochs);

        savemodel.save(@"./bias_regularizer_save_and_load", save_format: "tf");

        var loadModel = tf.keras.models.load_model(@"./bias_regularizer_save_and_load");
        loadModel.summary();

        loadModel.compile(tf.keras.optimizers.Adam(), tf.keras.losses.SparseCategoricalCrossentropy(from_logits: true), new string[] { "accuracy" });

        var fitDataset = new RandomDataSet(new Shape(227, 227, 3), 16);

        loadModel.fit(fitDataset.Data, fitDataset.Labels, batch_size, num_epochs);
    }


    [TestMethod]
    public void CreateConcatenateModelSaveAndLoad()
    {
        // a small demo model that is just here to see if the axis value for the concatenate method is saved and loaded.
        var input_layer = tf.keras.layers.Input((8, 8, 5));

        var conv1 = tf.keras.layers.Conv2D(2, kernel_size: 3, activation: "relu", padding: "same"/*, data_format: "_conv_1"*/).Apply(input_layer);
        conv1.Name = "conv1";

        var conv2 = tf.keras.layers.Conv2D(2, kernel_size: 3, activation: "relu", padding: "same"/*, data_format: "_conv_2"*/).Apply(input_layer);
        conv2.Name = "conv2";

        var concat1 = tf.keras.layers.Concatenate(axis: 3).Apply((conv1, conv2));
        concat1.Name = "concat1";

        var model = tf.keras.Model(input_layer, concat1);
        model.compile(tf.keras.optimizers.Adam(), tf.keras.losses.CategoricalCrossentropy());

        model.save(@"Assets/concat_axis3_model");

        
        var tensorInput = np.arange(320).reshape((1, 8, 8, 5)).astype(TF_DataType.TF_FLOAT);

        var tensors1 = model.predict(tensorInput);

        Assert.AreEqual((1, 8, 8, 4), tensors1.shape);

        model = null;
        keras.backend.clear_session();

        var model2 = tf.keras.models.load_model(@"Assets/concat_axis3_model");

        var tensors2 = model2.predict(tensorInput);

        Assert.AreEqual(tensors1.shape, tensors2.shape);
    }

}
