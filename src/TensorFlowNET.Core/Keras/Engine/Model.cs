using Tensorflow.Keras.Optimizers;

namespace Tensorflow.Keras.Engine
{
    public class Model : Network
    {
#pragma warning disable CS0169 // The field 'Model._cloning' is never used
        bool _cloning;
#pragma warning restore CS0169 // The field 'Model._cloning' is never used
#pragma warning disable CS0108 // Member hides inherited member; missing new keyword
        bool _is_compiled;
#pragma warning restore CS0108 // Member hides inherited member; missing new keyword
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
