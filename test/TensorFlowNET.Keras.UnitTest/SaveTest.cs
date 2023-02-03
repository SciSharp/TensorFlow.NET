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

namespace TensorFlowNET.Keras.UnitTest;

public static class AutoGraphExtension
{
    
}

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

        var g = ops.get_default_graph();

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
        
        model.save("C:\\Work\\tf.net\\tf_test\\tf.net.model", save_format:"pb");
    }

    [TestMethod]
    public void Temp()
    {
        var graph = new Graph();
        var g = graph.as_default();
        //var input_tensor = array_ops.placeholder(TF_DataType.TF_FLOAT, new int[] { 1 }, "test_string_tensor");
        var input_tensor = tf.placeholder(tf.int32, new int[] { 1 }, "aa");
        var wrapped_func = tf.autograph.to_graph(func);
        var res = wrapped_func(input_tensor);
        g.Exit();
    }

    private Tensor func(Tensor tensor)
    {
        return gen_ops.neg(tensor);
        //return array_ops.identity(tensor);
        //tf.device("cpu:0");
        //using (ops.control_dependencies(new object[] { res.op }))
        //{
        //    return array_ops.identity(tensor);
        //}
    }
}