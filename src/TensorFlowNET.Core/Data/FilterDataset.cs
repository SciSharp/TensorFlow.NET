using System;
using Tensorflow.Functions;
using static Tensorflow.Binding;

namespace Tensorflow
{
    /// <summary>
    /// A `Dataset` that filters its input according to a predicate function.
    /// </summary>
    public class FilterDataset : UnaryDataset
    {
        public FilterDataset(IDatasetV2 input_dataset,
            Func<Tensor, bool> predicate_func) : base(input_dataset)
        {
            Func<Tensors, Tensors> predicate_func_update = x =>
            {
                var result = predicate_func(x);
                return constant_op.constant(result);
            };

            var func = new ConcreteFunction($"{predicate_func.Method.Name}_{Tensorflow.ops.uid_function()}");
            func.Enter();
            var inputs = new Tensors();
            foreach (var input in input_dataset.element_spec)
                inputs.Add(tf.placeholder(input.dtype, shape: input.shape, name: "arg"));
            var outputs = predicate_func_update(inputs);
            func.ToGraph(inputs, outputs);
            func.Exit();

            structure = func.OutputStructure;

            variant_tensor = ops.filter_dataset(input_dataset.variant_tensor,
                func,
                output_types,
                output_shapes);
        }

        public FilterDataset(IDatasetV2 input_dataset,
            Func<Tensors, Tensors> predicate_func) : base(input_dataset)
        {
            var func = new ConcreteFunction($"{predicate_func.Method.Name}_{Tensorflow.ops.uid_function()}");
            func.Enter();
            var inputs = new Tensors();
            foreach (var input in input_dataset.element_spec)
                inputs.Add(tf.placeholder(input.dtype, shape: input.shape, name: "arg"));
            var outputs = predicate_func(inputs);
            func.ToGraph(inputs, outputs);
            func.Exit();

            structure = func.OutputStructure;

            variant_tensor = ops.filter_dataset(input_dataset.variant_tensor,
                func,
                output_types,
                output_shapes);
        }
    }
}
