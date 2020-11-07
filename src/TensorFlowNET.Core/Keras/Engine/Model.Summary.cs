using Tensorflow.Keras.Utils;

namespace Tensorflow.Keras.Engine
{
    public partial class Model
    {
        /// <summary>
        /// Prints a string summary of the network.
        /// </summary>
        public void summary(int line_length = -1, float[] positions = null)
        {
            layer_utils.print_summary(this,
                line_length: line_length,
                positions: positions);
        }
    }
}
