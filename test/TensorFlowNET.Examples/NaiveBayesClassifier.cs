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
            // separate training points by class
            // shape : nb_class * nb_samples * nb_features
            NDArray unique_y = y.unique<long>();
            NDArray points_by_class = np.array(y.Data<long>().Where(ys => unique_y.Data<long>().Contains(ys)));

            foreach (long cls in unique_y)
            {

            }


            // estimate mean and variance for each class / feature
            // shape : nb_classes * nb_features
            
        }
    }
}
