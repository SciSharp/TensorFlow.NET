using System.Collections.Generic;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.Saving.SavedModel;

public partial class KerasSavedModelUtils
{
    public static bool should_skip_serialization(object layer)
    {
        return false;
    }

    public static IDictionary<string, KerasObjectWrapper> wrap_layer_objects(Layer layer, object serialization_cache)
    {
        // TODO: process the loss

        return null;
    }
}