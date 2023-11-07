using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
using Tensorflow.Keras.Utils;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Layer that concatenates a list of inputs.
    /// </summary>
    public class Concatenate : Merge
    {
        MergeArgs args;
        int axis => args.Axis;

        public Concatenate(MergeArgs args) : base(args)
        {
            this.args = args;
        }

        public override void build(KerasShapesWrapper input_shape)
        {
            /*var shape_set = new HashSet<Shape>();
            var reduced_inputs_shapes = inputs.Select(x => x.shape).ToArray();
            for (var i = 0; i < reduced_inputs_shapes.Length; i++)
            {
                int seq = -1;
                Shape shape = reduced_inputs_shapes[i].Where(x =>
                {
                    seq++;
                    return seq != i;
                }).ToArray();
                shape_set.Add(shape);
            }*/
            _buildInputShape = input_shape;
            built = true;
        }

        protected override Tensors _merge_function(Tensors inputs)
        {
            return keras.backend.concatenate(inputs, axis: axis);
        }
    }
}
