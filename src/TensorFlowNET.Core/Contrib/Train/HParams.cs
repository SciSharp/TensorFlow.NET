namespace Tensorflow.Contrib.Train
{
    /// <summary>
    /// Class to hold a set of hyperparameters as name-value pairs.
    /// </summary>
    public class HParams
    {
        public bool load_pretrained { get; set; }

        public HParams(bool load_pretrained)
        {
            this.load_pretrained = load_pretrained;
        }
    }
}
