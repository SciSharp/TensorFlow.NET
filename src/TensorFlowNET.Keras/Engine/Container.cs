namespace Tensorflow.Keras.Engine
{
    public class Container
    {
        protected string[] _output_names;
        protected bool _built;

        public Container(string[] output_names)
        {
            _output_names = output_names;
        }
    }
}
