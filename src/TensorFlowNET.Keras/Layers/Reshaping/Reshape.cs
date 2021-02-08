using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using System.Collections.Generic;
using System;
using System.Linq;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Layer that reshapes inputs into the given shape.
    /// </summary>
    public class Reshape : Layer
    {
        ReshapeArgs args;
        public Reshape(ReshapeArgs args)
            : base(args)
        {
            this.args = args;
        }

        protected override Tensors Call(Tensors inputs, Tensor state = null, bool? training = null)
        {
            var shapes = new List<object>();
            shapes.Add(array_ops.shape(inputs)[0]);
            if (args.TargetShapeObjects != null)
                shapes.AddRange(args.TargetShapeObjects);
            if (args.TargetShape != null)
                args.TargetShape.dims.ToList().ForEach(x => shapes.Add(x));
            var shape = ops.convert_to_tensor(shapes);

            var result = array_ops.reshape(inputs, shape);
            if (!tf.Context.executing_eagerly())
                result.set_shape(ComputeOutputShape(inputs.shape));
            return result;
        }

        public override TensorShape ComputeOutputShape(TensorShape input_shape)
        {
            if (input_shape.dims[1..].Contains(-1))
            {
                throw new NotImplementedException("");
            }
            else
            {
                input_shape = input_shape.dims[0];
                var output_shape = input_shape.concatenate(args.TargetShape.dims);
                return output_shape;
            }
        }
    }
}
