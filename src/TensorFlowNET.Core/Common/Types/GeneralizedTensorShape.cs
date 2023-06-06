using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace Tensorflow.Common.Types
{
    public class GeneralizedTensorShape: IEnumerable<long?[]>
    {
        public TensorShapeConfig[] Shapes { get; set; }
        /// <summary>
        /// create a single-dim generalized Tensor shape.
        /// </summary>
        /// <param name="dim"></param>
        public GeneralizedTensorShape(int dim)
        {
            Shapes = new TensorShapeConfig[] { new TensorShapeConfig() { Items = new long?[] { dim } } };
        }

        public GeneralizedTensorShape(Shape shape)
        {
            Shapes = new TensorShapeConfig[] { shape };
        }

        public GeneralizedTensorShape(TensorShapeConfig shape)
        {
            Shapes = new TensorShapeConfig[] { shape };
        }

        public GeneralizedTensorShape(TensorShapeConfig[] shapes)
        {
            Shapes = shapes;
        }

        public GeneralizedTensorShape(IEnumerable<Shape> shape)
        {
            Shapes = shape.Select(x => (TensorShapeConfig)x).ToArray();
        }

        public Shape ToSingleShape()
        {
            if (Shapes.Length != 1)
            {
                throw new ValueError("The generalized shape contains more than 1 dim.");
            }
            var shape_config = Shapes[0];
            Debug.Assert(shape_config is not null);
            return new Shape(shape_config.Items.Select(x => x is null ? -1 : x.Value).ToArray());
        }

        public long ToNumber()
        {
            if(Shapes.Length != 1 || Shapes[0].Items.Length != 1)
            {
                throw new ValueError("The generalized shape contains more than 1 dim.");
            }
            var res = Shapes[0].Items[0];
            return res is null ? -1 : res.Value;
        }

        public Shape[] ToShapeArray()
        {
            return Shapes.Select(x => new Shape(x.Items.Select(y => y is null ? -1 : y.Value).ToArray())).ToArray();
        }

        public IEnumerator<long?[]> GetEnumerator()
        {
            foreach (var shape in Shapes)
            {
                yield return shape.Items;
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}
