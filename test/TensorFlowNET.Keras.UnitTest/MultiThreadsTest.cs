using System;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using System.Threading.Tasks;
using Tensorflow.NumPy;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace TensorFlowNET.Keras.UnitTest
{
    [TestClass]
    public class MultiThreads
    {
        [TestMethod]
        public void Test1()
        {
            //Arrange
            string savefile = "mymodel.h5";
            var model1 = BuildModel();
            model1.save_weights(savefile);
            var model2 = BuildModel();

            //act
            model1.load_weights(savefile);
            model2.load_weights(savefile);

        }

        [TestMethod]
        public void Test2()
        {
            //Arrange
            string savefile = "mymodel2.h5";
            var model1 = BuildModel();
            model1.save_weights(savefile);
            model1 = BuildModel(); //recreate model

            //act
            model1.load_weights(savefile);

        }

        [TestMethod]
        public void Test3Multithreading()
        {
            //Arrange
            string savefile = "mymodel3.h5";
            var model = BuildModel();
            model.save_weights(savefile);

            //Sanity check without multithreading
            for (int i = 0; i < 2; i++)
            {
                Functional clone = BuildModel();
                clone.load_weights(savefile);

                //Predict something
                clone.predict(np.array(new float[,] { { 0, 0 } }));
            } //works

            //act
            ParallelOptions parallelOptions = new ParallelOptions();
            parallelOptions.MaxDegreeOfParallelism = 8;
            var input = np.array(new float[,] { { 0, 0 } });
            Parallel.For(0, 8, parallelOptions, i =>
            {
                var clone = BuildModel();
                clone.load_weights(savefile);
                //Predict something
                clone.predict(input);
            });
        }

        Functional BuildModel()
        {
            tf.Context.reset_context();
            var inputs = keras.Input(shape: 2);

            // 1st dense layer
            var DenseLayer = keras.layers.Dense(1, activation: keras.activations.Sigmoid);
            var outputs = DenseLayer.Apply(inputs);

            // build keras model
            Functional model = keras.Model(inputs, outputs, name: Guid.NewGuid().ToString());
            // show model summary
            model.summary();

            // compile keras model into tensorflow's static graph
            model.compile(loss: keras.losses.MeanSquaredError(name: Guid.NewGuid().ToString()),
                optimizer: keras.optimizers.Adam(name: Guid.NewGuid().ToString()),
                metrics: new[] { "accuracy" });
            return model;
        }
    }
}
