using System;
using System.Collections.Generic;

namespace Tensorflow.Keras.Engine
{
    public partial class Layer
    {
        protected Dictionary<Layer, bool> trainable_state;
        protected Dictionary<Layer, bool> _compiled_trainable_state;

        /// <summary>
        /// Get the `trainable` state of each sublayer.
        /// </summary>
        /// <returns></returns>
        protected Dictionary<Layer, bool> _get_trainable_state()
        {
            trainable_state = new Dictionary<Layer, bool>();
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
