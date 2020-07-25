using System;
using System.Collections.Generic;
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
        /// 
        /// </summary>
        /// <param name="count"></param>
        /// <returns></returns>
        IDatasetV2 repeat(int count = -1);

        IDatasetV2 shuffle(int buffer_size, int? seed = null, bool reshuffle_each_iteration = true);

        IDatasetV2 batch(int batch_size, bool drop_remainder = false);

        IDatasetV2 prefetch(int buffer_size = -1, int? slack_period = null);

        IDatasetV2 take(int count);

        IDatasetV2 optimize(string[] optimizations, string[] optimization_configs);

        IDatasetV2 model(AutotuneAlgorithm algorithm, long cpu_budget);

        /// <summary>
        /// Apply options, such as optimization configuration, to the dataset.
        /// </summary>
        /// <returns></returns>
        IDatasetV2 apply_options();
    }
}
