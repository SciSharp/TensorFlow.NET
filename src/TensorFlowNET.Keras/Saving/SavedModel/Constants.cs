using System.Collections.Generic;

namespace Tensorflow.Keras.Saving.SavedModel;

public static class Constants
{
    /// <summary>
    /// Namespace used to store all attributes added during serialization.
    /// e.g. the list of layers can be accessed using `loaded.keras_api.layers`, in an
    /// object loaded from `tf.saved_model.load()`.
    /// </summary>
    public static readonly string KERAS_ATTR = "keras_api";
    /// <summary>
    /// Keys for the serialization cache.
    /// Maps to the keras serialization dict {Layer --> SerializedAttributes object}
    /// </summary>
    public static readonly string KERAS_CACHE_KEY = "keras_serialized_attributes";
    /// <summary>
    /// Name of Keras metadata file stored in the SavedModel.
    /// </summary>
    public static readonly string SAVED_METADATA_PATH = "keras_metadata.pb";
    
    public static readonly string INPUT_LAYER_IDENTIFIER = "_tf_keras_input_layer";
    public static readonly string LAYER_IDENTIFIER = "_tf_keras_layer";
    public static readonly string METRIC_IDENTIFIER = "_tf_keras_metric";
    public static readonly string MODEL_IDENTIFIER = "_tf_keras_model";
    public static readonly string NETWORK_IDENTIFIER = "_tf_keras_network";
    public static readonly string RNN_LAYER_IDENTIFIER = "_tf_keras_rnn_layer";
    public static readonly string SEQUENTIAL_IDENTIFIER = "_tf_keras_sequential";

    public static readonly IList<string> KERAS_OBJECT_IDENTIFIERS = new List<string>()
    {
        INPUT_LAYER_IDENTIFIER,
        LAYER_IDENTIFIER,
        METRIC_IDENTIFIER,
        MODEL_IDENTIFIER,
        NETWORK_IDENTIFIER,
        RNN_LAYER_IDENTIFIER,
        SEQUENTIAL_IDENTIFIER
    };
}
