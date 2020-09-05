using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// A `Dataset` that maps a function over elements in its input.
    /// </summary>
    public class MapDataset : UnaryDataset
    {
        public MapDataset(IDatasetV2 input_dataset,
            Func<Tensor, Tensor> map_func,
            bool use_inter_op_parallelism = true,
            bool preserve_cardinality = false,
            bool use_legacy_function = false) : base(input_dataset)
        {
            foreach(var input in input_dataset)
            {
                var data = map_func(input.Item1);
            }

            variant_tensor = ops.map_dataset(input_dataset.variant_tensor,
                output_types,
                output_shapes);
        }
    }
}
