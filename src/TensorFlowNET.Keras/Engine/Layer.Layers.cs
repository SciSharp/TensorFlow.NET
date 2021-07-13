using System;
using System.Collections.Generic;

namespace Tensorflow.Keras.Engine
{
    public partial class Layer
    {
        protected List<ILayer> _layers = new List<ILayer>();
        public List<ILayer> Layers => _layers;

        protected void StackLayers(params ILayer[] layers)
        {
            _layers.AddRange(layers);
        }

        public virtual Shape ComputeOutputShape(Shape input_shape)
            => throw new NotImplementedException("");
    }
}
