/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using NumSharp;
using System;
using Tensorflow;
using Tensorflow.Hub;
using static Tensorflow.Python;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// A nearest neighbor learning algorithm example
    /// This example is using the MNIST database of handwritten digits
    /// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/nearest_neighbor.py
    /// </summary>
    public class NearestNeighbor : IExample
    {
        public bool Enabled { get; set; } = true;
        public string Name => "Nearest Neighbor";
        Datasets<MnistDataSet> mnist;
        NDArray Xtr, Ytr, Xte, Yte;
        public int? TrainSize = null;
        public int ValidationSize = 5000;
        public int? TestSize = null;
        public bool IsImportingGraph { get; set; } = false;


        public bool Run()
        {
            // tf Graph Input
            var xtr = tf.placeholder(tf.float32, new TensorShape(-1, 784));
            var xte = tf.placeholder(tf.float32, new TensorShape(784));

            // Nearest Neighbor calculation using L1 Distance
            // Calculate L1 Distance
            var distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices: 1);
            // Prediction: Get min distance index (Nearest neighbor)
            var pred = tf.arg_min(distance, 0);

            float accuracy = 0f;
            // Initialize the variables (i.e. assign their default value)
            var init = tf.global_variables_initializer();
            using (var sess = tf.Session())
            {
                // Run the initializer
                sess.run(init);

                PrepareData();

                foreach(int i in range(Xte.shape[0]))
                {
                    // Get nearest neighbor
                    long nn_index = sess.run(pred, new FeedItem(xtr, Xtr), new FeedItem(xte, Xte[i]))[0];
                    // Get nearest neighbor class label and compare it to its true label
                    int index = (int)nn_index;

                    if (i % 10 == 0 || i == 0)
                        print($"Test {i} Prediction: {np.argmax(Ytr[index])} True Class: {np.argmax(Yte[i])}");

                    // Calculate accuracy
                    if ((int)np.argmax(Ytr[index]) == (int)np.argmax(Yte[i]))
                        accuracy += 1f/ Xte.shape[0];
                }

                print($"Accuracy: {accuracy}");
            }

            return accuracy > 0.8;
        }

        public void PrepareData()
        {
            mnist = MnistModelLoader.LoadAsync(".resources/mnist", oneHot: true, trainSize: TrainSize, validationSize: ValidationSize, testSize: TestSize, showProgressInConsole: true).Result;
            // In this example, we limit mnist data
            (Xtr, Ytr) = mnist.Train.GetNextBatch(TrainSize == null ? 5000 : TrainSize.Value / 100); // 5000 for training (nn candidates)
            (Xte, Yte) = mnist.Test.GetNextBatch(TestSize == null ? 200 : TestSize.Value / 100); // 200 for testing
        }

        public Graph ImportGraph()
        {
            throw new NotImplementedException();
        }

        public Graph BuildGraph()
        {
            throw new NotImplementedException();
        }

        public void Train(Session sess)
        {
            throw new NotImplementedException();
        }

        public void Predict(Session sess)
        {
            throw new NotImplementedException();
        }

        public void Test(Session sess)
        {
            throw new NotImplementedException();
        }
    }
}
