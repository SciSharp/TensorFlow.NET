using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Saving;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Abstract wrapper base class. Wrappers take another layer and augment it in various ways.
    /// Do not use this class as a layer, it is only an abstract base class.
    /// Two usable wrappers are the `TimeDistributed` and `Bidirectional` wrappers.
    /// </summary>
    public abstract class Wrapper: Layer
    {
        public ILayer _layer;
        public Wrapper(WrapperArgs args):base(args)
        {
            _layer = args.Layer;
        }

        public virtual void Build(KerasShapesWrapper input_shape)
        {
            if (!_layer.Built)
            {
                _layer.build(input_shape);
            }
            built = true;
        }

    }
}
