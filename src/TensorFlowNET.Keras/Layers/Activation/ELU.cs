using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers {
    /// <summary>
    /// ELU Layer:
    /// x = 0 when x > 0, x = alpha( e^x-1 ) elsewhere
    /// </summary>
    public class ELU : Layer
    {
        ELUArgs args;
        float alpha => args.Alpha;
        public ELU(ELUArgs args) : base(args)
        {
            this.args = args;
        }

        public override void build(KerasShapesWrapper input_shape)
        {
            if (alpha < 0f)
            {
                throw new ValueError("Alpha must be a number greater than 0.");
            }
            base.build(input_shape);
        }

        protected override Tensors Call(Tensors inputs, Tensor mask = null, bool? training = null, Tensors initial_state = null, Tensors constants = null)
        {
            Tensor output = inputs;
            output = tf.where(output > 0f, output,
                  tf.multiply(alpha, tf.sub(tf.exp(output), 1f)));
            return output;
        }
        public override Shape ComputeOutputShape(Shape input_shape)
        {
            return input_shape;
        }
    }
}
