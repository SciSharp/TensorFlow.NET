using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// A `Dataset` that repeats its input several times.
    /// </summary>
    public class RepeatDataset : UnaryUnchangedStructureDataset
    {
        Tensor _count;

        public RepeatDataset(IDatasetV2 input_dataset, int count = -1) :
            base(input_dataset)
        {
            _count = constant_op.constant(count, dtype: TF_DataType.TF_INT64, name: "count");
            variant_tensor = ops.repeat_dataset(input_dataset.variant_tensor,
                _count,
                input_dataset.output_types,
                input_dataset.output_shapes);
        }
    }
}
