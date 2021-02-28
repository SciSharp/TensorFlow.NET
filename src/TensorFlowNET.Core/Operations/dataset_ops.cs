using System;
using Tensorflow.Framework.Models;
using Tensorflow.Functions;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class dataset_ops
    {
        public Tensor tensor_dataset(Tensor[] components, TensorShape[] output_shapes, string name = null)
            => tf.Context.ExecuteOp("TensorDataset", name, new ExecuteOpArgs()
            {
                OpInputArgs = new object[] { components }
            }.SetAttributes(new { output_shapes }));

        /// <summary>
        /// Creates a dataset that emits each dim-0 slice of `components` once.
        /// </summary>
        /// <param name="components"></param>
        /// <param name="output_shapes"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor tensor_slice_dataset(Tensor[] components, TensorShape[] output_shapes, string name = null)
            => tf.Context.ExecuteOp("TensorSliceDataset", name, new ExecuteOpArgs()
            {
                OpInputArgs = new object[] { components }
            }.SetAttributes(new { output_shapes }));

        public Tensor range_dataset(Tensor start, Tensor stop, Tensor step, TF_DataType[] output_types, TensorShape[] output_shapes, string name = null)
            => tf.Context.ExecuteOp("RangeDataset", name, new ExecuteOpArgs(start, stop, step)
                .SetAttributes(new { output_types, output_shapes }));

        public Tensor repeat_dataset(Tensor input_dataset, Tensor count, TF_DataType[] output_types, TensorShape[] output_shapes, string name = null)
            => tf.Context.ExecuteOp("RepeatDataset", name, new ExecuteOpArgs(input_dataset, count)
                .SetAttributes(new { output_types, output_shapes }));

        public Tensor shard_dataset(Tensor input_dataset, Tensor num_shards, Tensor index,
            TF_DataType[] output_types, TensorShape[] output_shapes,
            bool require_non_empty = false, string name = null)
                => tf.Context.ExecuteOp("ShardDataset", name, new ExecuteOpArgs(input_dataset, num_shards, index)
                    .SetAttributes(new { require_non_empty, output_types, output_shapes }));

        public Tensor zip_dataset(Tensor[] input_datasets,
            TF_DataType[] output_types,
            TensorShape[] output_shapes,
            string name = null)
                => tf.Context.ExecuteOp("ZipDataset", name, new ExecuteOpArgs()
                {
                    OpInputArgs = new object[] { input_datasets }
                }.SetAttributes(new { output_types, output_shapes }));

        public Tensor shuffle_dataset_v3(Tensor input_dataset, Tensor buffer_size,
            Tensor seed, Tensor seed2, Tensor seed_generator,
            TF_DataType[] output_types, TensorShape[] output_shapes,
            bool reshuffle_each_iteration = true,
            string name = null)
                => tf.Context.ExecuteOp("ShuffleDatasetV3", name, new ExecuteOpArgs(input_dataset, buffer_size, seed, seed2, seed_generator)
                    .SetAttributes(new { reshuffle_each_iteration, output_types, output_shapes }));

        public Tensor skip_dataset(Tensor input_dataset, Tensor count,
            TF_DataType[] output_types, TensorShape[] output_shapes,
            string name = null)
                => tf.Context.ExecuteOp("SkipDataset", name, new ExecuteOpArgs(input_dataset, count)
                    .SetAttributes(new { output_types, output_shapes }));

        public Tensor dummy_seed_generator(string name = null)
            => tf.Context.ExecuteOp("DummySeedGenerator", name, new ExecuteOpArgs());

        public Tensor concatenate_dataset(Tensor input_dataset, Tensor another_dataset,
            TF_DataType[] output_types, TensorShape[] output_shapes,
            string name = null)
                => tf.Context.ExecuteOp("ConcatenateDataset", name, new ExecuteOpArgs(input_dataset, another_dataset)
                    .SetAttributes(new { output_types, output_shapes }));

        public Tensor cache_dataset_v2(Tensor input_dataset, Tensor filename, Tensor cache,
            TF_DataType[] output_types, TensorShape[] output_shapes,
            string name = null)
                => tf.Context.ExecuteOp("CacheDatasetV2", name, new ExecuteOpArgs(input_dataset, filename, cache)
                    .SetAttributes(new { output_types, output_shapes }));

        /// <summary>
        /// Creates a dataset that batches `batch_size` elements from `input_dataset`.
        /// </summary>
        /// <param name="input_dataset"></param>
        /// <param name="buffer_size"></param>
        /// <param name="drop_remainder"></param>
        /// <param name="output_types"></param>
        /// <param name="output_shapes"></param>
        /// <param name="parallel_copy"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor batch_dataset_v2(Tensor input_dataset, Tensor buffer_size,
            Tensor drop_remainder,
            TF_DataType[] output_types, TensorShape[] output_shapes,
            bool parallel_copy = false,
            string name = null)
                => tf.Context.ExecuteOp("BatchDatasetV2", name, 
                    new ExecuteOpArgs(input_dataset, buffer_size, drop_remainder)
                        .SetAttributes(new { parallel_copy, output_types, output_shapes }));

        /// <summary>
        /// 
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor dummy_memory_cache(string name = "")
            => tf.Context.ExecuteOp("DummyMemoryCache", name, new ExecuteOpArgs());

        /// <summary>
        /// Creates a dataset that asynchronously prefetches elements from `input_dataset`.
        /// </summary>
        /// <param name="input_dataset"></param>
        /// <param name="buffer_size"></param>
        /// <param name="output_types"></param>
        /// <param name="output_shapes"></param>
        /// <param name="slack_period"></param>
        /// <param name="legacy_autotune"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor prefetch_dataset(Tensor input_dataset, Tensor buffer_size,
            TF_DataType[] output_types, TensorShape[] output_shapes,
            int? slack_period = 0,
            bool legacy_autotune = true,
            string name = null)
                => tf.Context.ExecuteOp("PrefetchDataset", name, new ExecuteOpArgs(input_dataset, buffer_size)
                    .SetAttributes(new
                    {
                        output_types,
                        output_shapes,
                        slack_period,
                        legacy_autotune
                    }));

        /// <summary>
        /// Creates a dataset that contains `count` elements from the `input_dataset`.
        /// </summary>
        /// <param name="input_dataset"></param>
        /// <param name="count"></param>
        /// <param name="output_types"></param>
        /// <param name="output_shapes"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor take_dataset(Tensor input_dataset, Tensor count,
            TF_DataType[] output_types, TensorShape[] output_shapes,
            string name = null)
                => tf.Context.ExecuteOp("TakeDataset", name, new ExecuteOpArgs(input_dataset, count)
                    .SetAttributes(new { output_types, output_shapes }));

        /// <summary>
        /// Creates a dataset by applying optimizations to `input_dataset`.
        /// </summary>
        /// <param name="input_dataset"></param>
        /// <param name="optimizations"></param>
        /// <param name="output_types"></param>
        /// <param name="output_shapes"></param>
        /// <param name="optimization_configs"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor optimize_dataset(Tensor input_dataset, Tensor optimizations,
            TF_DataType[] output_types, TensorShape[] output_shapes,
            string[] optimization_configs = null,
            string name = null)
                => tf.Context.ExecuteOp("OptimizeDataset", name, new ExecuteOpArgs(input_dataset, optimizations)
                    .SetAttributes(new
                    {
                        output_types,
                        output_shapes,
                        optimization_configs = optimization_configs ?? new string[0]
                    }));

        public Tensor optimize_dataset_v2(Tensor input_dataset, Tensor optimizations_enabled,
            Tensor optimizations_disabled, Tensor optimizations_default,
            TF_DataType[] output_types, TensorShape[] output_shapes,
            string[] optimization_configs = null,
            string name = null)
                => tf.Context.ExecuteOp("OptimizeDatasetV2", name, new ExecuteOpArgs(input_dataset, 
                        optimizations_enabled, optimizations_disabled, optimizations_default)
                    .SetAttributes(new
                    {
                        output_types,
                        output_shapes,
                        optimization_configs = optimization_configs ?? new string[0]
                    }));

        /// <summary>
        /// Identity transformation that models performance.
        /// </summary>
        /// <param name="input_dataset"></param>
        /// <param name="output_types"></param>
        /// <param name="output_shapes"></param>
        /// <param name="algorithm"></param>
        /// <param name="cpu_budget"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor model_dataset(Tensor input_dataset,
            TF_DataType[] output_types, TensorShape[] output_shapes,
            AutotuneAlgorithm algorithm, long cpu_budget, long ram_budget,
            string name = null)
                => tf.Context.ExecuteOp("ModelDataset", name, new ExecuteOpArgs(input_dataset)
                    .SetAttributes(new
                    {
                        algorithm,
                        cpu_budget,
                        ram_budget,
                        output_types,
                        output_shapes
                    }));

        /// <summary>
        /// A container for an iterator resource.
        /// </summary>
        /// <param name="output_types"></param>
        /// <param name="output_shapes"></param>
        /// <param name="name"></param>
        /// <returns>A tuple of `Tensor` objects (handle, deleter).</returns>
        public (Tensor, Tensor) anonymous_iterator_v2(TF_DataType[] output_types, TensorShape[] output_shapes, string name = null)
        {
            var results = tf.Context.ExecuteOp("AnonymousIteratorV2", name, 
                new ExecuteOpArgs().SetAttributes(new { output_types, output_shapes }));
            return (results[0], results[1]);
        }

        /// <summary>
        /// Makes a new iterator from the given `dataset` and stores it in `iterator`.
        /// </summary>
        /// <param name="dataset"></param>
        /// <param name="iterator"></param>
        /// <param name="name"></param>
        /// <returns>The created Operation.</returns>
        public void make_iterator(Tensor dataset, Tensor iterator, string name = null)
            => tf.Context.ExecuteOp("MakeIterator", name, new ExecuteOpArgs(dataset, iterator));

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dataset"></param>
        /// <param name="iterator"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor map_dataset(Tensor dataset, ConcreteFunction f, TF_DataType[] output_types, TensorShape[] output_shapes,
            bool use_inter_op_parallelism = true, bool preserve_cardinality = false, string name = null)
                => tf.Context.ExecuteOp("MapDataset", name, new ExecuteOpArgs(dataset, new Tensor[0])
                    .SetAttributes(new
                    {
                        f,
                        output_types,
                        output_shapes,
                        use_inter_op_parallelism,
                        preserve_cardinality
                    }));

        /// <summary>
        /// Creates a dataset that applies `f` to the outputs of `input_dataset`.
        /// </summary>
        /// <param name="dataset"></param>
        /// <param name="f"></param>
        /// <param name="output_types"></param>
        /// <param name="output_shapes"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor flat_map_dataset(Tensor dataset, ConcreteFunction f, TF_DataType[] output_types, TensorShape[] output_shapes,
            string name = null)
                => tf.Context.ExecuteOp("FlatMapDataset", name, new ExecuteOpArgs(dataset, new Tensor[0])
                    .SetAttributes(new { f, output_types, output_shapes }));

        /// <summary>
        /// Creates a dataset that applies `f` to the outputs of `input_dataset`.
        /// </summary>
        /// <param name="dataset"></param>
        /// <param name="num_parallel_calls"></param>
        /// <param name="f"></param>
        /// <param name="output_types"></param>
        /// <param name="output_shapes"></param>
        /// <param name="use_inter_op_parallelism"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor parallel_map_dataset_v2(Tensor dataset, Tensor num_parallel_calls, ConcreteFunction f,
            TF_DataType[] output_types, TensorShape[] output_shapes,
            bool use_inter_op_parallelism = true,
            string deterministic = "default",
            bool preserve_cardinality = false,
            string name = null)
                => tf.Context.ExecuteOp("ParallelMapDatasetV2", name, 
                    new ExecuteOpArgs(dataset, new Tensor[0], num_parallel_calls)
                        .SetAttributes(new
                        {
                            f,
                            output_types,
                            output_shapes,
                            use_inter_op_parallelism,
                            deterministic,
                            preserve_cardinality
                        }));

        /// <summary>
        /// A container for an iterator resource.
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="deleter"></param>
        /// <param name="name"></param>
        /// <returns>The created Operation.</returns>
        public void delete_iterator(Tensor handle, Tensor deleter, string name = null)
            => tf.Context.ExecuteOp("DeleteIterator", name, new ExecuteOpArgs(handle, deleter));

        /// <summary>
        /// Gets the next output from the given iterator .
        /// </summary>
        /// <param name="iterator"></param>
        /// <param name="output_types"></param>
        /// <param name="output_shapes"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor[] iterator_get_next(Tensor iterator, TF_DataType[] output_types, TensorShape[] output_shapes, string name = null)
            => tf.Context.ExecuteOp("IteratorGetNext", name, new ExecuteOpArgs(iterator)
                .SetAttributes(new { output_types, output_shapes }));
    }
}
