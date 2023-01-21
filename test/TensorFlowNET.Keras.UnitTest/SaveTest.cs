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

namespace TensorFlowNET.Keras.UnitTest;

// class MNISTLoader
// {
//     public MNISTLoader()
//     {
//         var mnist = new MnistModelLoader()
//         
//     }
// }

[TestClass]
public class SaveTest
{
    [TestMethod]
    public void Test()
    {
        var inputs = new KerasInterface().Input((28, 28, 1));
        var x = new Flatten(new FlattenArgs()).Apply(inputs);
        x = new Dense(new DenseArgs() { Units = 100, Activation = tf.nn.relu }).Apply(x);
        x = new LayersApi().Dense(units: 10).Apply(x);
        var outputs = new LayersApi().Softmax(axis: 1).Apply(x);
        var model = new KerasInterface().Model(inputs, outputs);
        
        model.compile(new Adam(0.001f), new LossesApi().SparseCategoricalCrossentropy(), new  string[]{"accuracy"});

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
        
        model.save("", save_format:"pb");
    }
}