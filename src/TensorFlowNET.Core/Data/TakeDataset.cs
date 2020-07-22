using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class TakeDataset : UnaryUnchangedStructureDataset
    {
        Tensor _count;

        public TakeDataset(IDatasetV2 input_dataset, int count) :
            base(input_dataset)
        {
            _count = tf.convert_to_tensor(count, dtype: dtypes.int64, name: "count");
            variant_tensor = ops.take_dataset(input_dataset.variant_tensor, _count,
                output_types, output_shapes);
        }
    }
}
