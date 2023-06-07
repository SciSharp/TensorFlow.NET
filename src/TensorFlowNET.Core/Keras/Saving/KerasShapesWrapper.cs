using Newtonsoft.Json.Linq;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Text;
using System.Diagnostics;
using OneOf.Types;
using Tensorflow.Keras.Saving.Json;
using Tensorflow.Common.Types;

namespace Tensorflow.Keras.Saving
{
    [JsonConverter(typeof(CustomizedKerasShapesWrapperJsonConverter))]
    public class KerasShapesWrapper
    {
        public TensorShapeConfig[] Shapes { get; set; }

        public KerasShapesWrapper(Shape shape)
        {
            Shapes = new TensorShapeConfig[] { shape };
        }

        public KerasShapesWrapper(TensorShapeConfig shape)
        {
            Shapes = new TensorShapeConfig[] { shape };
        }

        public KerasShapesWrapper(TensorShapeConfig[] shapes)
        {
            Shapes = shapes;
        }

        public KerasShapesWrapper(IEnumerable<Shape> shape)
        {
            Shapes = shape.Select(x => (TensorShapeConfig)x).ToArray();
        }

        public Shape ToSingleShape()
        {
            Debug.Assert(Shapes.Length == 1);
            var shape_config = Shapes[0];
            Debug.Assert(shape_config is not null);
            return new Shape(shape_config.Items.Select(x => x is null ? -1 : x.Value).ToArray());
        }

        public Shape[] ToShapeArray()
        {
            return Shapes.Select(x => new Shape(x.Items.Select(y => y is null ? -1 : y.Value).ToArray())).ToArray();
        }

        public static implicit operator KerasShapesWrapper(Shape shape)
        {
            return new KerasShapesWrapper(shape);
        }
        public static implicit operator KerasShapesWrapper(TensorShapeConfig shape)
        {
            return new KerasShapesWrapper(shape);
        }

    }
}
