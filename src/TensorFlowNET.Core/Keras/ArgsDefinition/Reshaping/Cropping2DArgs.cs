using Tensorflow.NumPy;

namespace Tensorflow.Keras.ArgsDefinition.Reshaping
{
    public class Cropping2DArgs : LayerArgs
    {
        /// <summary>
        /// channel last: (b, h, w, c)
        /// channels_first: (b, c, h, w)
        /// </summary>
        public enum DataFormat { channels_first = 0, channels_last = 1 }
        /// <summary>
        /// Accept: int[1][2], int[1][1], int[2][2]
        /// </summary>
        public NDArray cropping { get; set; }
        public DataFormat data_format { get; set; } = DataFormat.channels_last;
    }
}
