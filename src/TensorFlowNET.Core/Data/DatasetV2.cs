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
        public string[] class_names { get; set; }
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

        public IDatasetV2 map(Func<Tensors, Tensors> map_func,
            bool use_inter_op_parallelism = true,
            bool preserve_cardinality = true,
            bool use_legacy_function = false)
            => new MapDataset(this,
                map_func,
                use_inter_op_parallelism: use_inter_op_parallelism,
                preserve_cardinality: preserve_cardinality,
                use_legacy_function: use_legacy_function);

        public IDatasetV2 map(Func<Tensors, Tensors> map_func, int num_parallel_calls)
            => new ParallelMapDataset(this, map_func, num_parallel_calls: num_parallel_calls);

        public OwnedIterator make_one_shot_iterator()
        {
            if (tf.Context.executing_eagerly())
            {
                // with ops.colocate_with(self._variant_tensor)
                return new OwnedIterator(this);
            }

            throw new NotImplementedException("");
        }

        public IDatasetV2 flat_map(Func<Tensor, IDatasetV2> map_func)
            => new FlatMapDataset(this, map_func);

        public IDatasetV2 model(AutotuneAlgorithm algorithm, long cpu_budget, long ram_budget)
            => new ModelDataset(this, algorithm, cpu_budget, ram_budget);

        public IDatasetV2 with_options(DatasetOptions options)
            => new OptionsDataset(this, options);

        public IDatasetV2 apply_options()
        {
            IDatasetV2 dataset = this;
            // (1) Apply threading options

            // (2) Apply autotune options
            var autotune = true;
            long cpu_budget = 0;
            long ram_budget = 0;
            if (autotune)
                dataset = dataset.model(AutotuneAlgorithm.HILL_CLIMB, cpu_budget, ram_budget);

            // (3) Apply graph rewrite options
            var graph_rewrites = new[]
            {
                "noop_elimination",
                "map_and_batch_fusion",
                "shuffle_and_repeat_fusion"
            };
            var graph_rewrite_configs = new string[]
            {
                "autotune_buffer_sizes:autotune:true",
                "disable_prefetch_legacy_autotune:autotune:true",
                "enable_gradient_descent:autotune:true",
                "map_parallelization:autotune:true"
            };

            dataset = new OptimizeDataset(dataset, new string[0], new string[0], graph_rewrites, graph_rewrite_configs);

            // (4) Apply stats aggregator options

            return dataset;
        }

        public Tensor dataset_cardinality(string name = null)
            => tf.Context.ExecuteOp("DatasetCardinality", name, new ExecuteOpArgs(variant_tensor));

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
