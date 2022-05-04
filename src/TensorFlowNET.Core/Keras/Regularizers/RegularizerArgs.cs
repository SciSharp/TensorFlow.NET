namespace Tensorflow.Keras
{
    public class RegularizerArgs
    {
        public Tensor X { get; set; }


        public RegularizerArgs(Tensor x)
        {
            X = x;          
        }
    }
}
