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
        public TensorSliceDataset(string[] array)
        {
            var element = tf.constant(array);
            _tensors = new[] { element };
            var batched_spec = new[] { element.ToTensorSpec() };
            structure = batched_spec.Select(x => x._unbatch()).ToArray();

            variant_tensor = ops.tensor_slice_dataset(_tensors, output_shapes);
        }

        public TensorSliceDataset(NDArray array)
        {
            var element = tf.constant(array);
            _tensors = new[] { element };
            var batched_spec = new[] { element.ToTensorSpec() };
            structure = batched_spec.Select(x => x._unbatch()).ToArray();

            variant_tensor = ops.tensor_slice_dataset(_tensors, output_shapes);
        }

        public TensorSliceDataset(Tensor tensor)
        {
            _tensors = new[] { tensor };
            var batched_spec = new[] { tensor.ToTensorSpec() };
            structure = batched_spec.Select(x => x._unbatch()).ToArray();

            variant_tensor = ops.tensor_slice_dataset(_tensors, output_shapes);
        }

        public TensorSliceDataset(Tensor features, Tensor labels)
        {
            _tensors = new[] { features, labels };
            var batched_spec = _tensors.Select(x => x.ToTensorSpec()).ToArray();
            structure = batched_spec.Select(x => x._unbatch()).ToArray();
            
            variant_tensor = ops.tensor_slice_dataset(_tensors, output_shapes);
        }
    }
}
