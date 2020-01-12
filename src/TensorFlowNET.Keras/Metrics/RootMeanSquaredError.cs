using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Metrics
{
    public class RootMeanSquaredError : Mean
    {
        public RootMeanSquaredError(string name = "root_mean_squared_error", string dtype = null)
            : base(name, dtype)
        {
        }
    }
}
