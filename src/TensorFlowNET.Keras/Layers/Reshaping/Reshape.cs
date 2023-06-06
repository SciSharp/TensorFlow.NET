using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using System.Collections.Generic;
using System;
using System.Linq;
using Tensorflow.Common.Types;

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

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            var shapes = new List<Tensor>();
            shapes.Add(array_ops.shape(inputs)[0]);
            var dtype = shapes[0].dtype;
            if (args.TargetShapeObjects != null)
                // shapes.AddRange(args.TargetShapeObjects);
                throw new NotImplementedException("");
            if (args.TargetShape != null)
                shapes.AddRange(args.TargetShape.dims.Select(x => constant_op.constant(x, dtype)));
            var shape = ops.convert_to_tensor(shapes);

            var result = array_ops.reshape(inputs, shape);
            if (!tf.Context.executing_eagerly())
                result.shape = ComputeOutputShape(inputs.shape);
            return result;
        }

        public override Shape ComputeOutputShape(Shape input_shape)
        {
            if (input_shape.dims.Skip(1).Contains(-1))
            {
                throw new NotImplementedException("");
            }
            else
            {
                input_shape = new Shape(input_shape.dims[0]);
                var output_shape = input_shape.concatenate(args.TargetShape.dims);
                return output_shape;
            }
        }
    }
}
