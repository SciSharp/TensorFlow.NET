using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Framework.Models;
using static Tensorflow.Binding;

namespace Tensorflow
{
    /// <summary>
    /// A `Dataset` that batches contiguous elements from its input.
    /// </summary>
    public class BatchDataset : UnaryDataset
    {
        Tensor _batch_size;
        Tensor _drop_remainder;

        public BatchDataset(IDatasetV2 input_dataset, int batch_size, bool drop_remainder = false) : 
            base(input_dataset)
        {
            _input_dataset = input_dataset;
            _batch_size = tf.convert_to_tensor(batch_size, dtype: TF_DataType.TF_INT64, name: "batch_size");
            _drop_remainder = tf.convert_to_tensor(drop_remainder, dtype: TF_DataType.TF_BOOL, name: "drop_remainder");
            
            if (drop_remainder)
            {
                throw new NotImplementedException("");
            }
            else
            {
                _structure = input_dataset.element_spec.Select(x => x._batch(-1)).ToArray();
            }

            variant_tensor = ops.batch_dataset_v2(input_dataset.variant_tensor,
                _batch_size,
                _drop_remainder,
                output_types,
                output_shapes);
        }
    }
}
