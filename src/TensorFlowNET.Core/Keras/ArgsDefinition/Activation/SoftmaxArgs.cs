using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.ArgsDefinition {
      public class SoftmaxArgs : LayerArgs {
            public Axis axis { get; set; } = -1;
      }
}
