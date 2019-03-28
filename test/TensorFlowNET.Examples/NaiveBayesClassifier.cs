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
        public int Priority => 100;
        public bool Enabled => true;
        public string Name => "Naive Bayes Classifier";

        public Normal dist { get; set; }
        public bool Run()
        {
            var X = np.array<double>(new double[][] { new double[] { 5.1, 3.5},new double[] { 4.9, 3.0 },new double[] { 4.7, 3.2 },
                                     new double[] { 4.6, 3.1 },new double[] { 5.0, 3.6 },new double[] { 5.4, 3.9 },
                                     new double[] { 4.6, 3.4 },new double[] { 5.0, 3.4 },new double[] { 4.4, 2.9 },
                                     new double[] { 4.9, 3.1 },new double[] { 5.4, 3.7 },new double[] {4.8, 3.4 },
                                     new double[] {4.8, 3.0 },new double[] {4.3, 3.0 },new double[] {5.8, 4.0 },
                                     new double[] {5.7, 4.4 },new double[] {5.4, 3.9 },new double[] {5.1, 3.5 },
                                     new double[] {5.7, 3.8 },new double[] {5.1, 3.8 },new double[] {5.4, 3.4 },
                                     new double[] {5.1, 3.7 },new double[] {5.1, 3.3 },new double[] {4.8, 3.4 },
                                     new double[] {5.0 , 3.0 },new double[] {5.0 , 3.4 },new double[] {5.2, 3.5 },
                                     new double[] {5.2, 3.4 },new double[] {4.7, 3.2 },new double[] {4.8, 3.1 },
                                     new double[] {5.4, 3.4 },new double[] {5.2, 4.1},new double[] {5.5, 4.2 },
                                     new double[] {4.9, 3.1 },new double[] {5.0 , 3.2 },new double[] {5.5, 3.5 },
                                     new double[] {4.9, 3.6 },new double[] {4.4, 3.0 },new double[] {5.1, 3.4 },
                                     new double[] {5.0 , 3.5 },new double[] {4.5, 2.3 },new double[] {4.4, 3.2 },
                                     new double[] {5.0 , 3.5 },new double[] {5.1, 3.8 },new double[] {4.8, 3.0},
                                     new double[] {5.1, 3.8 },new double[] {4.6, 3.2 },new double[] { 5.3, 3.7 },
                                     new double[] {5.0 , 3.3 },new double[] {7.0 , 3.2 },new double[] {6.4, 3.2 },
                                     new double[] {6.9, 3.1 },new double[] {5.5, 2.3 },new double[] {6.5, 2.8 },
                                     new double[] {5.7, 2.8 },new double[] {6.3, 3.3 },new double[] {4.9, 2.4 },
                                     new double[] {6.6, 2.9 },new double[] {5.2, 2.7 },new double[] {5.0 , 2.0 },
                                     new double[] {5.9, 3.0 },new double[] {6.0 , 2.2 },new double[] {6.1, 2.9 },
                                     new double[] {5.6, 2.9 },new double[] {6.7, 3.1 },new double[] {5.6, 3.0 },
                                     new double[] {5.8, 2.7 },new double[] {6.2, 2.2 },new double[] {5.6, 2.5 },
                                     new double[] {5.9, 3.0},new double[] {6.1, 2.8},new double[] {6.3, 2.5},
                                     new double[] {6.1, 2.8},new double[] {6.4, 2.9},new double[] {6.6, 3.0 },
                                     new double[] {6.8, 2.8},new double[] {6.7, 3.0 },new double[] {6.0 , 2.9},
                                     new double[] {5.7, 2.6},new double[] {5.5, 2.4},new double[] {5.5, 2.4},
                                     new double[] {5.8, 2.7},new double[] {6.0 , 2.7},new double[] {5.4, 3.0 },
                                     new double[] {6.0 , 3.4},new double[] {6.7, 3.1},new double[] {6.3, 2.3},
                                     new double[] {5.6, 3.0 },new double[] {5.5, 2.5},new double[] {5.5, 2.6},
                                     new double[] {6.1, 3.0 },new double[] {5.8, 2.6},new double[] {5.0 , 2.3},
                                     new double[] {5.6, 2.7},new double[] {5.7, 3.0 },new double[] {5.7, 2.9},
                                     new double[] {6.2, 2.9},new double[] {5.1, 2.5},new double[] {5.7, 2.8},
                                     new double[] {6.3, 3.3},new double[] {5.8, 2.7},new double[] {7.1, 3.0 },
                                     new double[] {6.3, 2.9},new double[] {6.5, 3.0 },new double[] {7.6, 3.0 },
                                     new double[] {4.9, 2.5},new double[] {7.3, 2.9},new double[] {6.7, 2.5},
                                     new double[] {7.2, 3.6},new double[] {6.5, 3.2},new double[] {6.4, 2.7},
                                     new double[] {6.8, 3.00 },new double[] {5.7, 2.5},new double[] {5.8, 2.8},
                                     new double[] {6.4, 3.2},new double[] {6.5, 3.0 },new double[] {7.7, 3.8},
                                     new double[] {7.7, 2.6},new double[] {6.0 , 2.2},new double[] {6.9, 3.2},
                                     new double[] {5.6, 2.8},new double[] {7.7, 2.8},new double[] {6.3, 2.7},
                                     new double[] {6.7, 3.3},new double[] {7.2, 3.2},new double[] {6.2, 2.8},
                                     new double[] {6.1, 3.0 },new double[] {6.4, 2.8},new double[] {7.2, 3.0 },
                                     new double[] {7.4, 2.8},new double[] {7.9, 3.8},new double[] {6.4, 2.8},
                                     new double[] {6.3, 2.8},new double[] {6.1, 2.6},new double[] {7.7, 3.0 },
                                     new double[] {6.3, 3.4},new double[] {6.4, 3.1},new double[] {6.0, 3.0},
                                     new double[] {6.9, 3.1},new double[] {6.7, 3.1},new double[] {6.9, 3.1},
                                     new double[] {5.8, 2.7},new double[] {6.8, 3.2},new double[] {6.7, 3.3},
                                     new double[] {6.7, 3.0 },new double[] {6.3, 2.5},new double[] {6.5, 3.0 },
                                     new double[] {6.2, 3.4},new double[] {5.9, 3.0 }, new double[] {5.8, 3.0 }});
            
            var y = np.array<int>(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
            fit(X, y);
            // Create a regular grid and classify each point 
            double x_min = (double) X.amin(0)[0] - 0.5;
            double y_min = (double) X.amin(0)[1] - 0.5;
            double x_max = (double) X.amax(0)[0] + 0.5;
            double y_max = (double) X.amax(0)[1] + 0.5;

            var (xx, yy) = np.meshgrid(np.linspace(x_min, x_max, 30), np.linspace(y_min, y_max, 30));
            var s = tf.Session();
            var samples = np.vstack(xx.ravel(), yy.ravel());
            var Z = s.run(predict(samples));


            return true;
        }

        public void fit(NDArray X, NDArray y)
        {
            NDArray unique_y = y.unique<long>();

            Dictionary<long, List<List<double>>> dic = new Dictionary<long, List<List<double>>>();
            // Init uy in dic
            foreach (int uy in unique_y.Data<int>())
            {
                dic.Add(uy, new List<List<double>>());
            }
            // Separate training points by class 
            // Shape : nb_classes * nb_samples * nb_features
            int maxCount = 0;
            for (int i = 0; i < y.size; i++)
            {
                long curClass = (long)y[i];
                List<List<double>> l = dic[curClass];
                List<double> pair = new List<double>();
                pair.Add((double)X[i,0]);
                pair.Add((double)X[i, 1]);
                l.Add(pair);
                if (l.Count > maxCount)
                {
                    maxCount = l.Count;
                }
                dic[curClass] = l;
            }
            double[,,] points = new double[dic.Count, maxCount, X.shape[1]];
            foreach (KeyValuePair<long, List<List<double>>> kv in dic)
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
            NDArray points_by_class = np.array<double>(points);
            // estimate mean and variance for each class / feature
            // shape : nb_classes * nb_features
            var cons = tf.constant(points_by_class);
            var tup = tf.nn.moments(cons, new int[]{1});
            var mean = tup.Item1;
            var variance = tup.Item2;

            // Create a 3x2 univariate normal distribution with the 
            // Known mean and variance           
            var dist = tf.distributions.Normal(mean, tf.sqrt(variance));
            this.dist = dist;
        }

        public Tensor predict (NDArray X)
        {
            if (dist == null)
            {
                throw new ArgumentNullException("cant not find the model (normal distribution)!");
            }
            int nb_classes = (int) dist.scale().shape[0];
            int nb_features = (int)dist.scale().shape[1];

            // Conditional probabilities log P(x|c) with shape
            // (nb_samples, nb_classes)
            var t1= ops.convert_to_tensor(X, TF_DataType.TF_DOUBLE);
            //var t2 = ops.convert_to_tensor(new int[] { 1, nb_classes });
            //Tensor tile = tf.tile(t1, t2);
            Tensor tile = tf.tile(X, new int[] { 1, nb_classes });
            Tensor r = tf.reshape(tile, new Tensor(new int[] { -1, nb_classes, nb_features }));
            var cond_probs = tf.reduce_sum(dist.log_prob(r));
            // uniform priors
            var priors = np.log(np.array<double>((1.0 / nb_classes) * nb_classes));

            // posterior log probability, log P(c) + log P(x|c)
            var joint_likelihood = tf.add(new Tensor(priors), cond_probs);
            // normalize to get (log)-probabilities

            var norm_factor = tf.reduce_logsumexp(joint_likelihood, new int[] { 1 }, true);
            var log_prob = joint_likelihood - norm_factor;
            // exp to get the actual probabilities
            return tf.exp(log_prob);
        }

        public void PrepareData()
        {
            
        }
    }
}
