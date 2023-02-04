using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Multiply inputs by `scale` and adds `offset`.
    /// </summary>
    public class Rescaling : Layer
    {
        RescalingArgs args;
        Tensor scale;
        Tensor offset;

        public Rescaling(RescalingArgs args) : base(args)
        {
            this.args = args;
        }

        protected override Tensors Call(Tensors inputs, Tensor state = null, bool? training = null)
        {
            scale = constant_op.constant(args.Scale, args.DType);
            offset = constant_op.constant(args.Offset, args.DType);
            return math_ops.cast(inputs, args.DType) * scale + offset;
        }

        public override Shape ComputeOutputShape(Shape input_shape)
        {
            return input_shape;
        }
    }
}
