using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class DatasetManager
    {
        public IDatasetV2 from_tensor_slices(NDArray features, NDArray labels)
            => new TensorSliceDataset(features, labels);
    }
}
