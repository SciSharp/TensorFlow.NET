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
           // t/f.nn.moments()
        }

        public void fit(NDArray X, NDArray y)
        {
            NDArray unique_y = y.unique<long>();
            
            Dictionary<int, List<NDArray>> dic = new Dictionary<int, List<NDArray>>();
            // Init uy in dic
            foreach (int uy in unique_y.Data<int>())
            {
                dic.Add(uy, new List<NDArray>());
            }
            // Separate training points by class 
            // Shape : nb_classes * nb_samples * nb_features
            int maxCount = 0;
            foreach (var (x, t) in zip(X.Data<float>(), y.Data<int>()))
            {
                int curClass = (y[t, 0] as NDArray).Data<int>().First();
                List<NDArray> l = dic[curClass];
                l.Add(x);
                if (l.Count > maxCount)
                {
                    maxCount = l.Count;
                }
                dic.Add(curClass, l);
            }
            NDArray points_by_class = np.zeros(dic.Count,maxCount,X.shape[1]);
            foreach (KeyValuePair<int, List<NDArray>> kv in dic)
            {
                var cls = kv.Value.ToArray();
                for (int i = 0; i < dic.Count; i++)
                {
                    points_by_class[i] = dic[i];
                }
            }

            // estimate mean and variance for each class / feature
            // shape : nb_classes * nb_features
            var cons = tf.constant(points_by_class);
        }
    }
}
