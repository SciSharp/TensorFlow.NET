using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;

namespace Tensorflow.Keras.Metrics
{
    /// <summary>
    /// Computes the (weighted) mean of the given values.
    /// </summary>
    public class Mean : Reduce
    {
        public Mean(string name = "mean", TF_DataType dtype = TF_DataType.DtInvalid)
            : base(Reduction.WEIGHTED_MEAN, name, dtype: dtype)
        {

        }
    }
}
