using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow.NumPy;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Metrics;
using Tensorflow.Keras.Optimizers;
using Tensorflow.Operations;

namespace TensorFlowNET.Keras.UnitTest.SaveModel;

[TestClass]
public class SequentialModelTest
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

        model.save("C:\\Work\\tf.net\\tf_test\\tf.net.simple.compile", save_format: "tf");
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

        model.save("C:\\Work\\tf.net\\tf_test\\tf.net.simple.sequential", save_format: "tf");
    }
}