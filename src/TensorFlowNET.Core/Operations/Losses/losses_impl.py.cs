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
                    (labels, logits, weights) = _remove_squeezable_dimensions(
        labels, logits, weights, expected_rank_diff: 1);

                });

            throw new NotImplementedException("sparse_softmax_cross_entropy");
        }

        public (Tensor, Tensor, float) _remove_squeezable_dimensions(Tensor labels,
            Tensor predictions,
            float weights = 0,
            int expected_rank_diff = 0)
        {
            (labels, predictions, weights) = confusion_matrix.remove_squeezable_dimensions(
                labels, predictions, expected_rank_diff: expected_rank_diff);

            throw new NotImplementedException("_remove_squeezable_dimensions");
        }
    }
}
