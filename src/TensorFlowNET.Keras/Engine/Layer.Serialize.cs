using System.Collections.Generic;
using System.Linq;
using Tensorflow.Keras.Saving.SavedModel;
using Tensorflow.Train;

namespace Tensorflow.Keras.Engine;

public abstract partial class Layer
{
    public LayerSavedModelSaver TrackableSavedModelSaver => new LayerSavedModelSaver(this);

    public string ObjectIdentifier => TrackableSavedModelSaver.ObjectIdentifier;

    public string TrackingMetadata => TrackableSavedModelSaver.TrackingMetadata;

    public override IDictionary<string, Trackable> _trackable_children(SaveType save_type = SaveType.CHECKPOINT, IDictionary<string, object>? cache = null)
    {
        IDictionary<string, Trackable> children;
        if (save_type == SaveType.SAVEDMODEL)
        {
            // TODO: deal with cache.
            children = TrackableSavedModelSaver.trackable_children(cache);
        }
        else
        {
            children = new Dictionary<string, Trackable>();
        }

        return children.Concat(base._trackable_children(save_type, cache)).ToDictionary(x => x.Key, x => x.Value);
    }
}