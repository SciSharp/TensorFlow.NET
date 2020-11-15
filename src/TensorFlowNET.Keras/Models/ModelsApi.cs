using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;

namespace Tensorflow.Keras.Models
{
    public class ModelsApi
    {
        public Functional from_config(ModelConfig config)
            => Functional.from_config(config);
    }
}
