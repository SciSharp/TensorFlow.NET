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

namespace TensorFlowNET.Keras.UnitTest.SaveModel;

[TestClass]
public class SequentialModelLoad
{
    [TestMethod]
    public void SimpleModelFromSequential()
    {
        //new SequentialModelSave().SimpleModelFromSequential();
        var model = keras.models.load_model(@"D:\development\tf.net\tf_test\tf.net.simple.sequential");

        model.summary();

        model.compile(new Adam(0.0001f), new LossesApi().SparseCategoricalCrossentropy(), new string[] { "accuracy" });

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
        model.summary();
    }
}
