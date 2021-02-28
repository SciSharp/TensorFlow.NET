using System;
using System.Collections.Generic;
using Tensorflow.Framework.Models;

namespace Tensorflow
{
    public interface IDatasetV2 : IEnumerable<(Tensor, Tensor)>
    {
        string[] class_names { get; set; }

        Tensor variant_tensor { get; set; }

        TensorShape[] output_shapes { get; }

        TF_DataType[] output_types { get; }

        TensorSpec[] element_spec { get; }

        TensorSpec[] structure { get; set; }

        /// <summary>
        /// Caches the elements in this dataset.
        /// </summary>
        /// <param name="filename"></param>
        /// <returns></returns>
        IDatasetV2 cache(string filename = "");

        /// <summary>
        /// Creates a `Dataset` by concatenating the given dataset with this dataset.
        /// </summary>
        /// <param name="dataset"></param>
        /// <returns></returns>
        IDatasetV2 concatenate(IDatasetV2 dataset);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="count"></param>
        /// <returns></returns>
        IDatasetV2 repeat(int count = -1);

        /// <summary>
        /// Creates a `Dataset` that includes only 1/`num_shards` of this dataset.
        /// </summary>
        /// <param name="num_shards">The number of shards operating in parallel</param>
        /// <param name="index">The worker index</param>
        /// <returns></returns>
        IDatasetV2 shard(int num_shards, int index);

        IDatasetV2 shuffle(int buffer_size, int? seed = null, bool reshuffle_each_iteration = true);

        /// <summary>
        /// Creates a `Dataset` that skips `count` elements from this dataset.
        /// </summary>
        /// <param name="count"></param>
        /// <returns></returns>
        IDatasetV2 skip(int count);

        IDatasetV2 batch(int batch_size, bool drop_remainder = false);

        IDatasetV2 prefetch(int buffer_size = -1, int? slack_period = null);

        IDatasetV2 take(int count);

        IDatasetV2 optimize(string[] optimizations, string[] optimization_configs);

        IDatasetV2 map(Func<Tensors, Tensors> map_func,
            bool use_inter_op_parallelism = true,
            bool preserve_cardinality = true,
            bool use_legacy_function = false);

        IDatasetV2 map(Func<Tensors, Tensors> map_func,
            int num_parallel_calls);

        OwnedIterator make_one_shot_iterator();

        IDatasetV2 flat_map(Func<Tensor, IDatasetV2> map_func);

        IDatasetV2 model(AutotuneAlgorithm algorithm, long cpu_budget, long ram_budget);

        IDatasetV2 with_options(DatasetOptions options);

        /// <summary>
        /// Apply options, such as optimization configuration, to the dataset.
        /// </summary>
        /// <returns></returns>
        IDatasetV2 apply_options();

        /// <summary>
        /// Returns the cardinality of `dataset`, if known.
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        Tensor dataset_cardinality(string name = null);
    }
}
