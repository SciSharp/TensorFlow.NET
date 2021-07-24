﻿using Tensorflow.NumPy;

namespace Tensorflow.Keras.ArgsDefinition {
      public class Cropping3DArgs : LayerArgs {
            /// <summary>
            /// channel last: (b, h, w, c)
            /// channels_first: (b, c, h, w)
            /// </summary>
            public enum DataFormat { channels_first = 0, channels_last = 1 }
            /// <summary>
            /// Accept: int[1][3], int[1][1], int[3][2]
            /// </summary>
            public NDArray cropping { get; set; }
            public DataFormat data_format { get; set; } = DataFormat.channels_last;
      }
}
