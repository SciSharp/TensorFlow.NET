using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Framework;
using Tensorflow.Framework.Models;
using static Tensorflow.Binding;

namespace Tensorflow.Data
{
    /// <summary>
    /// A `Dataset` that concatenates its input with given dataset.
    /// </summary>
    public class ConcatenateDataset : DatasetV2
    {
        IDatasetV2 _input_dataset;
        IDatasetV2 _dataset_to_concatenate;
        public ConcatenateDataset(IDatasetV2 input_dataset, IDatasetV2 dataset_to_concatenate)
        {
            _input_dataset = input_dataset;
            _dataset_to_concatenate = dataset_to_concatenate;
            var _structure = new List<TensorSpec>();
            foreach(var (i, spec) in enumerate(dataset_to_concatenate.element_spec))
            {
                var shape = _input_dataset.output_shapes[i].most_specific_compatible_shape(spec.shape);
                _structure.Add(new TensorSpec(shape, dtype: spec.dtype));
            }
            structure = _structure.ToArray();

            variant_tensor = ops.concatenate_dataset(input_dataset.variant_tensor, 
                    dataset_to_concatenate.variant_tensor,
                    output_types, output_shapes);
        }
    }
}
