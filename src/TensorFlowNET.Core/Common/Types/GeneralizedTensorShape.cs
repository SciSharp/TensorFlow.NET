using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace Tensorflow.Common.Types
{
    public class GeneralizedTensorShape: Nest<Shape>
    {
        public GeneralizedTensorShape(Shape value, string? name = null)
        {
            NodeValue = value;
            NestType = NestType.Node;
        }

        public GeneralizedTensorShape(IEnumerable<Shape> values, string? name = null)
        {
            ListValue = values.Select(s => new Nest<Shape>(s) as INestStructure<Shape>).ToList();
            Name = name;
            NestType = NestType.List;
        }

        public GeneralizedTensorShape(Dictionary<string, Shape> value, string? name = null)
        {
            DictValue = value.ToDictionary(x => x.Key, x => new Nest<Shape>(x.Value) as INestStructure<Shape>);
            Name = name;
            NestType = NestType.Dictionary;
        }

        public GeneralizedTensorShape(Nest<Shape> other)
        {
            NestType = other.NestType;
            NodeValue = other.NodeValue;
            DictValue = other.DictValue;
            ListValue = other.ListValue;
            Name = other.Name;
        }

        public Shape ToSingleShape()
        {
            var shapes = Flatten().ToList();
            if (shapes.Count != 1)
            {
                throw new ValueError("The generalized shape contains more than 1 dim.");
            }
            return shapes[0];
        }

        public long ToNumber()
        {
            var shapes = Flatten().ToList();
            if (shapes.Count != 1 || shapes[0].ndim != 1)
            {
                throw new ValueError("The generalized shape contains more than 1 dim.");
            }
            return shapes[0].dims[0];
        }

        public INestStructure<TensorShapeConfig> ToTensorShapeConfigs()
        {
            return MapStructure(s => new TensorShapeConfig() { Items = s.dims.Select<long, long?>(x => x == -1 ? null : x).ToArray() });
        }

        public static implicit operator GeneralizedTensorShape(Shape shape)
        {
            return new GeneralizedTensorShape(shape);
        }
    }
}
