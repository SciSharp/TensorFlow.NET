using System;
using System.Collections.Generic;

namespace Tensorflow.Keras.Engine
{
    public partial class Layer
    {
        public virtual List<ILayer> Layers => _self_tracked_trackables;

        protected void StackLayers(params ILayer[] layers)
        {
            _self_tracked_trackables.AddRange(layers);
        }

        public virtual Shape ComputeOutputShape(Shape input_shape)
            => throw new NotImplementedException("");
    }
}
