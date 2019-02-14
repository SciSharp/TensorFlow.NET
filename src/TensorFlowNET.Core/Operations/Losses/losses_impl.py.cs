using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class LossesImpl : Python
    {
        public Tensor sparse_softmax_cross_entropy(Tensor labels, 
            Tensor logits,
            float weights = 1.0f,
            string scope = "",
            string loss_collection= "losses")
        {
            with<ops.name_scope>(new ops.name_scope(scope,
                "sparse_softmax_cross_entropy_loss",
                (logits, labels, weights)),
                namescope =>
                {


                });

            throw new NotImplementedException("sparse_softmax_cross_entropy");
        }
    }
}
