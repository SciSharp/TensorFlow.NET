using System;
using System.Collections.Generic;

namespace Tensorflow.Keras.Engine
{
    public partial class Layer
    {
        protected Dictionary<ILayer, bool> trainable_state;
        protected Dictionary<ILayer, bool> _compiled_trainable_state;

        /// <summary>
        /// Get the `trainable` state of each sublayer.
        /// </summary>
        /// <returns></returns>
        protected Dictionary<ILayer, bool> _get_trainable_state()
        {
            trainable_state = new Dictionary<ILayer, bool>();
            foreach (var layer in _flatten_layers())
                trainable_state[layer] = layer.Trainable;
            return trainable_state;
        }

        void _set_trainable_state(Dictionary<Layer, object> trainable_state)
        {
            throw new NotImplementedException("");
        }
    }
}
