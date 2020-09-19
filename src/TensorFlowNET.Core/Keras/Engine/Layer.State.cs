using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Engine
{
    public partial class Layer
    {
        Dictionary<Layer, object> trainable_state;
        Dictionary<Layer, object> _get_trainable_state()
        {
            trainable_state = new Dictionary<Layer, object>();
            throw new NotImplementedException("");
        }

        void _set_trainable_state(Dictionary<Layer, object> trainable_state)
        {
            throw new NotImplementedException("");
        }
    }
}
