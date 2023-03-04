using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.Models
{
    public interface IModelsApi
    {
        public IModel load_model(string filepath, bool compile = true, LoadOptions? options = null);
    }
}
