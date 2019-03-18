using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Layers;

namespace Tensorflow.Keras.Engine
{
    public class Sequential : Model, IPython
    {
        public Sequential(string name = null) 
            : base(name: name)
        {
            supports_masking = true;
            _compute_output_and_mask_jointly = true;
        }

        public void __enter__()
        {
            
        }

        public void add(Layer layer)
        {
            built = false;
            var set_inputs = false;
        }

        public void __exit__()
        {
            
        }

        public void Dispose()
        {

        }
    }
}
