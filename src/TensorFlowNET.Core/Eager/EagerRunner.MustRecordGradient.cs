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

        private bool ShouldRecord(Tensor[] inputs)
        {
            bool should_record = false;
            foreach (var tape in tf.GetTapeSet())
            {
                if (tape.ShouldRecord(inputs))
                {
                    should_record = true;
                    break;
                }
            }
            return should_record;
        }
    }
}
