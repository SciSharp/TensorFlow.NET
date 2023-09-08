using Tensorflow.Keras.Saving;
using Tensorflow.Keras.Saving.SavedModel;

namespace Tensorflow.Keras.Models;

public class ModelsApi: IModelsApi
{
    public Functional from_config(FunctionalConfig config)
        => Functional.from_config(config);

    public IModel load_model(string filepath, bool compile = true, LoadOptions? options = null)
    {
        return KerasLoadModelUtils.load_model(filepath, compile: compile, options: options) as Model;
    }
}
