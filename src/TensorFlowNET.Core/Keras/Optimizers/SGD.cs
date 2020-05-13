using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Optimizers
{
    public class SGD : OptimizerV2
    {
        protected override string _name => "SGD";
        
        bool nesterov;

        public SGD(float learning_rate, 
            float momentum = 0.0f,
            bool nesterov = false,
            float decay = 0.0f) : base()
        {
            _set_hyper("learning_rate", learning_rate);
            _set_hyper("decay", decay);

            _momentum = momentum > 0;

            _set_hyper("momentum", momentum);

            nesterov = nesterov;
        }
    }
}
