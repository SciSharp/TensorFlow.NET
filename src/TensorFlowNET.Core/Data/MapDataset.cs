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
            Func<Tensors, Tensors> map_func,
            bool use_inter_op_parallelism = true,
            bool preserve_cardinality = false,
            bool use_legacy_function = false) : base(input_dataset)
        {
            var func = new ConcreteFunction($"{map_func.Method.Name}_{Tensorflow.ops.uid_function()}");
            func.Enter();
            var inputs = new Tensors();
            foreach (var input in input_dataset.element_spec)
                inputs.Add(tf.placeholder(input.dtype, shape: input.shape, name: "arg"));
            var outputs = map_func(inputs);
            func.ToGraph(inputs, outputs);
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
