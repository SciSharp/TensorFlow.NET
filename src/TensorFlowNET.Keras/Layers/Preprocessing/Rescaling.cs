using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Common.Types;

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

<<<<<<< HEAD
<<<<<<< HEAD
        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
=======
        protected override Tensors Call(Tensors inputs, Tensor mask = null, bool? training = null, Tensors initial_state = null, Tensors constants = null)
>>>>>>> master
=======
        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
>>>>>>> 90a65d7d98b92f26574ac32392ed802a57d4d2c8
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
