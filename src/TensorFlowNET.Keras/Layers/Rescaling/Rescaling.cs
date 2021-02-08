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
            scale = math_ops.cast(args.Scale, args.DType);
            offset = math_ops.cast(args.Offset, args.DType);
            return math_ops.cast(inputs, args.DType) * scale + offset;
        }

        public override TensorShape ComputeOutputShape(TensorShape input_shape)
        {
            return input_shape;
        }
    }
}
