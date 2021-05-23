/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using System;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class embedding_ops
    {
        /// <summary>
        /// Helper function for embedding_lookup and _compute_sampled_logits.
        /// </summary>
        /// <param name="params"></param>
        /// <param name="ids"></param>
        /// <param name="partition_strategy"></param>
        /// <param name="name"></param>
        /// <param name="max_norm"></param>
        /// <returns></returns>
        public static Tensor _embedding_lookup_and_transform(IVariableV1 @params,
            Tensor ids,
            string partition_strategy = "mod",
            string name = null,
            string max_norm = null)
        {
            return tf_with(ops.name_scope(name, "embedding_lookup", new { @params, ids }), scope =>
            {
                name = scope;
                int np = 1;
                ids = ops.convert_to_tensor(ids, name: "ids");
                if (np == 1)
                {
                    var gather = array_ops.gather(@params.AsTensor(), ids, name: name);
                    var result = _clip(gather, ids, max_norm);

                    return array_ops.identity(result);
                }

                throw new NotImplementedException("_embedding_lookup_and_transform");
            });
        }

        public static Tensor _embedding_lookup_and_transform(Tensor[] @params,
                Tensor ids,
                string partition_strategy = "mod",
                string name = null,
                string max_norm = null)
        {
            return tf_with(ops.name_scope(name, "embedding_lookup", new { @params, ids }), scope =>
            {
                name = scope;
                int np = @params.Length;
                @params = ops.convert_n_to_tensor_or_indexed_slices(@params, name: "params");
                ids = ops.convert_to_tensor(ids, name: "ids");
                if (np == 1)
                {
                    ops.colocate_with(@params[0]);
                    var result = _clip(array_ops.gather(@params[0], ids, name: name), ids, max_norm);
                    return array_ops.identity(result);
                }
                else
                {
                    // Flatten the ids. There are two cases where we need to do this.
                    throw new NotImplementedException("_embedding_lookup_and_transform");
                }
            });
        }

        public static Tensor _clip(Tensor @params, Tensor ids, string max_norm = null)
        {
            if (max_norm == null)
                return @params;

            throw new NotImplementedException("_clip");
        }

        public static Tensor embedding_lookup(Tensor[] @params, Tensor ids,
            string partition_strategy = "mod",
            string name = null,
            bool validate_indices = true,
            string max_norm = null)
        {
            return _embedding_lookup_and_transform(@params: @params,
              ids: ids,
              partition_strategy: partition_strategy,
              name: name,
              max_norm: max_norm);
        }

        public static Tensor embedding_lookup(IVariableV1 @params, Tensor ids,
            string partition_strategy = "mod",
            string name = null,
            bool validate_indices = true,
            string max_norm = null)
        {
            return _embedding_lookup_and_transform(@params: @params,
              ids: ids,
              partition_strategy: partition_strategy,
              name: name,
              max_norm: max_norm);
        }
    }
}
