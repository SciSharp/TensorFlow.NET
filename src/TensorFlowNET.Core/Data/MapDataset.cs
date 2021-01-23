using System;
using Tensorflow.Functions;
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
            var func = new ConcreteFunction($"{map_func.Method.Name}_{Guid.NewGuid()}");
            func.Enter();
            var input = tf.placeholder(input_dataset.element_spec[0].dtype);
            var output = map_func(input);
            func.ToGraph(input, output);
            func.Exit();

            structure = func.OutputStructure;

            variant_tensor = ops.map_dataset(input_dataset.variant_tensor,
                func,
                output_types,
                output_shapes,
                use_inter_op_parallelism: use_inter_op_parallelism,
                preserve_cardinality: preserve_cardinality);
        }
    }
}
