using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class TensorSliceDataset : IDatasetV2
    {
        NDArray features;
        NDArray labels;

        public TensorSliceDataset(NDArray features, NDArray labels)
        {
            this.features = features;
            this.labels = labels;
        }
    }
}
