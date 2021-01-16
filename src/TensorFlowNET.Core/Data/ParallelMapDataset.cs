using System;
using System.Linq;
using Tensorflow.Functions;
using static Tensorflow.Binding;

namespace Tensorflow
{
    //A `Dataset` that maps a function over elements in its input in parallel.
    public class ParallelMapDataset : UnaryDataset
    {
        public ParallelMapDataset(IDatasetV2 input_dataset,
            Func<Tensors, Tensors> map_func,
            int num_parallel_calls = -1,
            bool use_inter_op_parallelism = true,
            bool preserve_cardinality = false,
            bool use_legacy_function = false) : base(input_dataset)
        {
            var func = new ConcreteFunction(map_func,
                input_dataset.element_spec.Select(x => x.dtype).ToArray(),
                input_dataset.element_spec.Select(x => x.shape).ToArray());

            structure = func.OutputStructure;
            var _num_parallel_calls = tf.convert_to_tensor(num_parallel_calls, dtype: tf.int64,
                name: "num_parallel_calls");
            variant_tensor = ops.parallel_map_dataset_v2(input_dataset.variant_tensor,
                _num_parallel_calls,
                func,
                output_types,
                output_shapes);
        }
    }
}
