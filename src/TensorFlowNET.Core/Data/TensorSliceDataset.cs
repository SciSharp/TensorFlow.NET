using NumSharp;
using NumSharp.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Framework.Models;
using static Tensorflow.Binding;

namespace Tensorflow.Data
{
    public class TensorSliceDataset : DatasetSource
    {
        public TensorSliceDataset(Tensor features, Tensor labels)
        {
            _tensors = new[] { features, labels };
            var batched_spec = _tensors.Select(x => x.ToTensorSpec()).ToArray();
            structure = batched_spec.Select(x => x._unbatch()).ToArray();
            
            variant_tensor = ops.tensor_slice_dataset(_tensors, output_shapes);
        }
    }
}
