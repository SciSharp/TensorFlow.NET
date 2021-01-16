using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Data;
using Tensorflow.Framework.Models;
using static Tensorflow.Binding;

namespace Tensorflow
{
    /// <summary>
    /// Abstract class representing a dataset with no inputs.
    /// </summary>
    public class DatasetV2 : IDatasetV2
    {
        protected dataset_ops ops = new dataset_ops();
        public Tensor variant_tensor { get; set; }

        public TensorSpec[] structure { get; set; }

        public TensorShape[] output_shapes => structure.Select(x => x.shape).ToArray();

        public TF_DataType[] output_types => structure.Select(x => x.dtype).ToArray();

        public TensorSpec[] element_spec => structure;

        public IDatasetV2 cache(string filename = "")
            => new CacheDataset(this, filename: filename);

        public IDatasetV2 concatenate(IDatasetV2 dataset)
            => new ConcatenateDataset(this, dataset);

        public IDatasetV2 take(int count = -1)
            => new TakeDataset(this, count: count);

        public IDatasetV2 batch(int batch_size, bool drop_remainder = false)
            => new BatchDataset(this, batch_size, drop_remainder: drop_remainder);

        public IDatasetV2 prefetch(int buffer_size = -1, int? slack_period = null)
            => new PrefetchDataset(this, buffer_size: buffer_size, slack_period: slack_period);

        public IDatasetV2 repeat(int count = -1)
            => new RepeatDataset(this, count: count);

        public IDatasetV2 shard(int num_shards, int index)
            => new ShardDataset(this, num_shards, index);

        public IDatasetV2 shuffle(int buffer_size, int? seed = null, bool reshuffle_each_iteration = true)
            => new ShuffleDataset(this, buffer_size, seed: seed, reshuffle_each_iteration: reshuffle_each_iteration);

        public IDatasetV2 skip(int count)
            => new SkipDataset(this, count);

        public IDatasetV2 optimize(string[] optimizations, string[] optimization_configs)
            => new OptimizeDataset(this, optimizations, optimization_configs: optimization_configs);

        public IDatasetV2 map(Func<Tensor, Tensor> map_func,
            bool use_inter_op_parallelism = true,
            bool preserve_cardinality = true,
            bool use_legacy_function = false)
            => new MapDataset(this,
                map_func,
                use_inter_op_parallelism: use_inter_op_parallelism,
                preserve_cardinality: preserve_cardinality,
                use_legacy_function: use_legacy_function);

        public IDatasetV2 map(Func<Tensors, Tensors> map_func, int num_parallel_calls = -1)
            => new ParallelMapDataset(this, map_func, num_parallel_calls: num_parallel_calls);

        public IDatasetV2 flat_map(Func<Tensor, IDatasetV2> map_func)
            => new FlatMapDataset(this, map_func);

        public IDatasetV2 model(AutotuneAlgorithm algorithm, long cpu_budget)
            => new ModelDataset(this, algorithm, cpu_budget);

        public IDatasetV2 with_options(DatasetOptions options)
            => new OptionsDataset(this, options);

        public IDatasetV2 apply_options()
        {
            // (1) Apply threading options
            var graph_rewrites = new[]
            {
                "map_and_batch_fusion",
                "noop_elimination",
                "shuffle_and_repeat_fusion"
            };

            var graph_rewrite_configs = new string[0];

            // (2) Apply graph rewrite options
            var dataset = optimize(graph_rewrites, graph_rewrite_configs);

            // (3) Apply autotune options
            var autotune = true;
            long cpu_budget = 0;

            if (autotune)
                dataset = dataset.model(AutotuneAlgorithm.HILL_CLIMB, cpu_budget);

            // (4) Apply stats aggregator options

            return dataset;
        }

        public Tensor dataset_cardinality(string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "DatasetCardinality", name,
                    null,
                    variant_tensor);
                return results[0];
            }

            throw new NotImplementedException("");
        }

        public override string ToString()
            => $"{GetType().Name} shapes: {string.Join(", ", structure.Select(x => x.shape))}, types: {string.Join(", ", structure.Select(x => "tf." + x.dtype.as_numpy_name()))}";

        public IEnumerator<(Tensor, Tensor)> GetEnumerator()
        {
            using var ownedIterator = new OwnedIterator(this);

            Tensor[] results = null;
            while (true)
            {
                try
                {
                    results = ownedIterator.next();
                }
                catch (StopIteration)
                {
                    break;
                }

                yield return (results[0], results.Length == 1 ? null : results[1]);
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }
    }
}
