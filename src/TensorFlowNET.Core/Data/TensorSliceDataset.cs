using NumSharp;
using NumSharp.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Framework.Models;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class TensorSliceDataset : DatasetSource
    {
        public TensorSliceDataset(NDArray features, NDArray labels)
        {
            _tensors = new[] { tf.convert_to_tensor(features), tf.convert_to_tensor(labels) };
            var batched_spec = _tensors.Select(x => x.ToTensorSpec()).ToArray();
            _structure = batched_spec.Select(x => x._unbatch()).ToArray();
            
            variant_tensor = ops.tensor_slice_dataset(_tensors, output_shapes);
        }
    }
}
