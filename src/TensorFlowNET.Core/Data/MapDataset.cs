using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Functions;
using Tensorflow.Graphs;
using static Tensorflow.Binding;

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
            var func = new ConcreteFunction(map_func, input_dataset.element_spec[0].dtype);

            variant_tensor = ops.map_dataset(input_dataset.variant_tensor,
                func,
                output_types,
                output_shapes);
        }
    }
}
