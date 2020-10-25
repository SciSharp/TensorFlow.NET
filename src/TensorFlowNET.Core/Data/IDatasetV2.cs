using System;
using System.Collections.Generic;
using System.Linq.Expressions;
using System.Text;
using Tensorflow.Framework.Models;

namespace Tensorflow
{
    public interface IDatasetV2 : IEnumerable<(Tensor, Tensor)>
    {
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
        IDatasetV2 cache(string filename="");

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

        IDatasetV2 map(Func<Tensor, Tensor> map_func, 
            bool use_inter_op_parallelism = true,
            bool preserve_cardinality = false,
            bool use_legacy_function = false);

        IDatasetV2 flat_map(Func<Tensor, IDatasetV2> map_func);

        IDatasetV2 model(AutotuneAlgorithm algorithm, long cpu_budget);

        /// <summary>
        /// Apply options, such as optimization configuration, to the dataset.
        /// </summary>
        /// <returns></returns>
        IDatasetV2 apply_options();
    }
}
