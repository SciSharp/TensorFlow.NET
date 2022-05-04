namespace Tensorflow.Keras
{
    public class L2 : IRegularizer
    {
        float l2;

        public L2(float l2 = 0.01f)
        {
            this.l2 = l2;
        }

        public Tensor Apply(RegularizerArgs args)
        {
            return l2 * math_ops.reduce_sum(math_ops.square(args.X));
        }
    }
}
