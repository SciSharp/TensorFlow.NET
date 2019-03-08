using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class embedding_ops : Python
    {
        /// <summary>
        /// Helper function for embedding_lookup and _compute_sampled_logits.
        /// </summary>
        /// <param name="params"></param>
        /// <param name="ids"></param>
        /// <param name="partition_strategy"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor _embedding_lookup_and_transform(RefVariable @params,
                Tensor ids,
                string partition_strategy = "mod",
                string name = null,
                string max_norm = null)
        {
            return with(new ops.name_scope(name, "embedding_lookup", new { @params, ids }), scope =>
            {
                name = scope;
                int np = 1;
                ids = ops.convert_to_tensor(ids, name: "ids");
                if(np == 1)
                {
                    var gather = array_ops.gather(@params, ids, name: name);
                    var result = _clip(gather, ids, max_norm);

                    return array_ops.identity(result);
                }

                throw new NotImplementedException("_embedding_lookup_and_transform");
            });
        }

        public static Tensor _clip(Tensor @params, Tensor ids, string max_norm = null)
        {
            if (max_norm == null)
                return @params;

            throw new NotImplementedException("_clip");
        }
    }
}
