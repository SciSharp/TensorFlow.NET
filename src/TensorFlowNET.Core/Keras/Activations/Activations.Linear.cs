namespace Tensorflow.Keras
{
    public partial class Activations
    {
        /// <summary>
        /// Linear activation function (pass-through).
        /// </summary>
        public Activation Linear = (features, name) => features;
    }
}
