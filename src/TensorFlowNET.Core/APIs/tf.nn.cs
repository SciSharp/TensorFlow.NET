using System;
using System.Collections.Generic;
using System.Text;
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
        }
    }
}
