using Tensorflow.Keras.Optimizers;

namespace Tensorflow.Keras.Engine
{
    public class Model : Network
    {
        bool _cloning;
        bool _is_compiled;
        string loss;
        IOptimizer optimizer;

        public Model(string name = null) 
            : base(name: name)
        {

        }

        public void compile(string optimizerName, string lossName)
        {
            switch (optimizerName)
            {
                case "rmsprop":
                    optimizer = new RMSprop();
                    break;
            }

            loss = lossName;
            _is_compiled = true;

            // Prepare list of loss functions, same size of model outputs.
        }
    }
}
