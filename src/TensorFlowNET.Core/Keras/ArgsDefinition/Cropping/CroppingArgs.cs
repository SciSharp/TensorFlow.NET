using Tensorflow.NumPy;

namespace Tensorflow.Keras.ArgsDefinition {
      public class CroppingArgs : LayerArgs {
            /// <summary>
            /// Accept length 1 or 2
            /// </summary>
            public NDArray cropping { get; set; }
      }
}