using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow
{
    /// <summary>
    /// A `Dataset` for sharding its input.
    /// </summary>
    public class ShardDataset : UnaryUnchangedStructureDataset
    {
        Tensor _num_shards;
        Tensor _index;

        public ShardDataset(IDatasetV2 input_dataset,
            int num_shards,
            int index) : base(input_dataset)
        {
            _num_shards = tf.convert_to_tensor(num_shards, dtype: TF_DataType.TF_INT64, name: "num_shards");
            _index = tf.convert_to_tensor(index, dtype: TF_DataType.TF_INT64, name: "index");

            variant_tensor = ops.shard_dataset
                (input_dataset.variant_tensor,
                num_shards: _num_shards,
                index: _index,
                input_dataset.output_types,
                input_dataset.output_shapes);
        }
    }
}
