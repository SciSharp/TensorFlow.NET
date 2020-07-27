using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Optimizers;

namespace Tensorflow.Keras.Engine
{
    /// <summary>
    /// `Model` groups layers into an object with training and inference features.
    /// </summary>
    public class Model : Layer
    {
#pragma warning disable CS0169 // The field 'Model._cloning' is never used
        bool _cloning;
#pragma warning restore CS0169 // The field 'Model._cloning' is never used
#pragma warning disable CS0108 // Member hides inherited member; missing new keyword
#pragma warning disable CS0414 // The field 'Model._is_compiled' is assigned but its value is never used
        bool _is_compiled;
#pragma warning restore CS0414 // The field 'Model._is_compiled' is assigned but its value is never used
#pragma warning restore CS0108 // Member hides inherited member; missing new keyword
        string loss;
        IOptimizer optimizer;

        public Model(ModelArgs args) 
            : base(args)
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
