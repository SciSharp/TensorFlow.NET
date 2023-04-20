using System;
using Tensorflow.Gradients;
using static Tensorflow.Binding;
using static Tensorflow.tensorflow;

namespace Tensorflow.Eager
{
    public partial class EagerRunner
    {
        public bool MustRecordGradient()
        {
            return HasGradientTape();
        }

        public int TFE_TapeSetPossibleGradientTypes(Tensor[] tensors)
        {
            var tape_set = tf.GetTapeSet();
            var input_ids = MakeTensorIDList(tensors);
            var input_dtypes = MakeTensorDtypeList(tensors);
            bool some_tape_watching = false;
            if (tape_set is not null && tape_set.Count > 0)
            {
                foreach (var tape in tape_set)
                {
                    if (tape.ShouldRecord(input_ids, input_dtypes))
                    {
                        if (tape.Persistent || some_tape_watching)
                        {
                            return gradients_util.POSSIBLE_GRADIENT_TYPES_HIGHER_ORDER;
                        }
                        some_tape_watching = true;
                    }
                }
            }
            // skip the forward_accumulators.

            if (some_tape_watching)
            {
                return gradients_util.POSSIBLE_GRADIENT_TYPES_FIRST_ORDER;
            }
            else
            {
                return gradients_util.POSSIBLE_GRADIENT_TYPES_NONE;
            }
        }
    }
}
