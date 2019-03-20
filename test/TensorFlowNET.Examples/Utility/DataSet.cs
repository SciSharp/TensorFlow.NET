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
        private int _epochs_completed;
        private int _index_in_epoch;
        private NDArray _images;
        private NDArray _labels;

        public DataSet(NDArray images, NDArray labels, TF_DataType dtype, bool reshape)
        {
            _num_examples = images.shape[0];
            images = images.reshape(images.shape[0], images.shape[1] * images.shape[2]);
            images.astype(dtype.as_numpy_datatype());
            images = np.multiply(images, 1.0f / 255.0f);

            _images = images;
            _labels = labels;
            _epochs_completed = 0;
            _index_in_epoch = 0;
        }
    }
}
