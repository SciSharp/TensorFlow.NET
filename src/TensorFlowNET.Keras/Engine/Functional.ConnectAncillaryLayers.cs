using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Saving;
using Tensorflow.Keras.Utils;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Engine
{
    public partial class Functional
    {
        /// <summary>
        /// Adds layers that are not connected to the outputs to the model.
        /// </summary>
        /// <param name="created_layers"></param>
        public void connect_ancillary_layers(Dictionary<string, ILayer> created_layers)
        {
            
        }
    }
}
