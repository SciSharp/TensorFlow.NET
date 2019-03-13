using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations;
using Tensorflow.Operations.Activation;

namespace Tensorflow
{
    public static partial class tf
    {
        public static class nn
        {
            public static (Tensor, Tensor) moments(Tensor x,
                int[] axes,
                string name = null,
                bool keep_dims = false) => nn_impl.moments(x, 
                    axes, 
                    name: name, 
                    keep_dims: keep_dims);

            public static Tensor embedding_lookup(RefVariable @params,
                Tensor ids,
                string partition_strategy = "mod",
                string name = null) => embedding_ops._embedding_lookup_and_transform(@params,
                    ids,
                    partition_strategy: partition_strategy,
                    name: name);

            public static IActivation relu => new relu();

            public static Tensor[] fused_batch_norm(Tensor x,
                RefVariable scale,
                RefVariable offset,
                Tensor mean = null,
                Tensor variance = null,
                float epsilon = 0.001f,
                string data_format = "NHWC",
                bool is_training = true,
                string name = null) => nn_impl.fused_batch_norm(x, scale, offset, mean, variance,
                    epsilon: epsilon,
                    data_format: data_format,
                    is_training: is_training,
                    name: name);

            public static Tensor max_pool() => gen_nn_ops.max_pool();
        }
    }
}
