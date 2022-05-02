using System;

namespace Tensorflow.Keras
{
    public class L1 : IRegularizer
    {
        float l1;

        public L1(float l1 = 0.01f)
        {
            this.l1 = l1;
        }

        public Tensor Apply(RegularizerArgs args)
        {
            return l1 * math_ops.reduce_sum(math_ops.abs(args.X));
        }
    }
}
