using Tensorflow.NumPy;

namespace Tensorflow.Keras.ArgsDefinition.Reshaping
{
    public class Cropping1DArgs : LayerArgs
    {
        /// <summary>
        /// Accept length 1 or 2
        /// </summary>
        public NDArray cropping { get; set; }
    }
}