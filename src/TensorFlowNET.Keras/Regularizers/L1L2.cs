using System;
using static Tensorflow.Binding;
namespace Tensorflow.Keras
{
    public class L1L2 : IRegularizer
    {
        float l1;
        float l2;

        public L1L2(float l1 = 0.0f, float l2 = 0.0f)
        {
            this.l1 = l1;
            this.l2 = l2;

        }
        public Tensor Apply(RegularizerArgs args)
        {
            Tensor regularization = tf.constant(0.0, args.X.dtype);
            regularization += l1 * math_ops.reduce_sum(math_ops.abs(args.X));
            regularization += l2 * math_ops.reduce_sum(math_ops.square(args.X));
            return regularization;
        }
    }
}
