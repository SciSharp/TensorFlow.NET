using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using NumSharp.Core;
using System.Linq;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// https://github.com/nicolov/naive_bayes_tensorflow
    /// </summary>
    public class NaiveBayesClassifier : Python, IExample
    { 
        public void Run()
        {
            np.array<float>(1.0f, 1.0f);
            // var X = np.array<float>(np.array<float>(1.0f, 1.0f), np.array<float>(2.0f, 2.0f), np.array<float>(1.0f, -1.0f), np.array<float>(2.0f, -2.0f), np.array<float>(-1.0f, -1.0f), np.array<float>(-1.0f, 1.0f),);
            // var X = np.array<float[]>(new float[][] { new float[] { 1.0f, 1.0f}, new float[] { 2.0f, 2.0f }, new float[] { -1.0f, -1.0f }, new float[] { -2.0f, -2.0f }, new float[] { 1.0f, -1.0f }, new float[] { 2.0f, -2.0f }, });
            var X = np.array<float>(new float[][] { new float[] { 1.0f, 1.0f }, new float[] { 2.0f, 2.0f }, new float[] { -1.0f, -1.0f }, new float[] { -2.0f, -2.0f }, new float[] { 1.0f, -1.0f }, new float[] { 2.0f, -2.0f }, });
            var y = np.array<int>(0,0,1,1,2,2);
            fit(X, y);
            // Create a regular grid and classify each point 
            
        }

        public void fit(NDArray X, NDArray y)
        {
            NDArray unique_y = y.unique<long>();

            Dictionary<long, List<List<float>>> dic = new Dictionary<long, List<List<float>>>();
            // Init uy in dic
            foreach (int uy in unique_y.Data<int>())
            {
                dic.Add(uy, new List<List<float>>());
            }
            // Separate training points by class 
            // Shape : nb_classes * nb_samples * nb_features
            int maxCount = 0;
            for (int i = 0; i < y.size; i++)
            {
                long curClass = (long)y[i];
                List<List<float>> l = dic[curClass];
                List<float> pair = new List<float>();
                pair.Add((float)X[i,0]);
                pair.Add((float)X[i, 1]);
                l.Add(pair);
                if (l.Count > maxCount)
                {
                    maxCount = l.Count;
                }
                dic[curClass] = l;
            }
            float[,,] points = new float[dic.Count, maxCount, X.shape[1]];
            foreach (KeyValuePair<long, List<List<float>>> kv in dic)
            {
                int j = (int) kv.Key;
                for (int i = 0; i < maxCount; i++)
                {
                    for (int k = 0; k < X.shape[1]; k++)
                    {
                        points[j, i, k] = kv.Value[i][k];
                    }
                }

            }
            NDArray points_by_class = np.array<float>(points);
            // estimate mean and variance for each class / feature
            // shape : nb_classes * nb_features
            var cons = tf.constant(points_by_class);
            var tup = tf.nn.moments(cons, new int[]{1});
            var mean = tup.Item1;
            var variance = tup.Item2;

            // Create a 3x2 univariate normal distribution with the 
            // Known mean and variance           
            var dist = tf.distributions.Normal(mean, tf.sqrt(variance));

        }

        public void predict (NDArray X)
        {
            // assert self.dist is not None
            // nb_classes, nb_features = map(int, self.dist.scale.shape)


            throw new NotFiniteNumberException();
        }
    }
}
