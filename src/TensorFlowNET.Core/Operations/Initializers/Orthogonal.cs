using System;
using System.Linq;
using static Tensorflow.TensorShapeProto.Types;

namespace Tensorflow.Operations.Initializers
{
    public class Orthogonal : IInitializer
    {
        float _gain = 0f;

        public Orthogonal(float gain = 1.0f, int? seed = null)
        {

        }

        public Tensor Apply(InitializerArgs args)
        {
            return _generate_init_val(args.Shape, args.DType);
        }

        private Tensor _generate_init_val(Shape shape, TF_DataType dtype)
        {
            var num_rows = 1L;
            foreach (var dim in shape.dims.Take(shape.ndim - 1))
                num_rows *= dim;
            var num_cols = shape.dims.Last();
            var flat_shape = (Math.Max(num_cols, num_rows), Math.Min(num_cols, num_rows));

            throw new NotImplementedException("");
        }
    }
}
