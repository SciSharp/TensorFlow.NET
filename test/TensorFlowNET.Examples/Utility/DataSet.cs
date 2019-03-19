using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.Examples.Utility
{
    public class DataSet
    {
        private int _num_examples;

        public DataSet(NDArray images, NDArray labels, TF_DataType dtype, bool reshape)
        {
            _num_examples = images.shape[0];
            images = images.reshape(images.shape[0], images.shape[1] * images.shape[2]);
            images = np.multiply(images, 1.0f / 255.0f);
        }
    }
}
