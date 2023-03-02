using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving.SavedModel;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Metrics;
using Tensorflow;
using Tensorflow.Keras.Optimizers;
using static Tensorflow.KerasApi;
using Tensorflow.NumPy;
using static TensorFlowNET.Keras.UnitTest.SaveModel.SequentialModelSave;

namespace TensorFlowNET.Keras.UnitTest.SaveModel;

[TestClass]
public class SequentialModelLoad
{
    [TestMethod]
    public void SimpleModelFromAutoCompile()
    {
        var model = keras.models.load_model(@"Assets/simple_model_from_auto_compile");
        model.summary();

        model.compile(new Adam(0.0001f), new LossesApi().SparseCategoricalCrossentropy(), new string[] { "accuracy" });

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
            ValidationSize = 50000,
        }).Result;

        model.fit(dataset.Train.Data, dataset.Train.Labels, batch_size, num_epochs);
    }

    [TestMethod]
    public void AlexnetFromSequential()
    {
        new SequentialModelSave().AlexnetFromSequential();
        var model = keras.models.load_model(@"./alexnet_from_sequential");
        model.summary();

        model.compile(new Adam(0.001f), new LossesApi().SparseCategoricalCrossentropy(from_logits: true), new string[] { "accuracy" });

        var num_epochs = 1;
        var batch_size = 8;

        var dataset = new RandomDataSet(new Shape(227, 227, 3), 16);

        model.fit(dataset.Data, dataset.Labels, batch_size, num_epochs);
    }
}
