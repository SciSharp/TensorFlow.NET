using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Tensorflow.Keras.Saving.SavedModel;
using Tensorflow.Train;

namespace Tensorflow.Keras.Engine;

public abstract partial class Layer
{
    public virtual SavedModelSaver TrackableSavedModelSaver => new LayerSavedModelSaver(this);

    public override string ObjectIdentifier => TrackableSavedModelSaver.ObjectIdentifier;

    public string GetTrackingMetadata() => TrackableSavedModelSaver.TrackingMetadata;

    public override IDictionary<string, Trackable> _trackable_children(SaveType save_type = SaveType.CHECKPOINT, IDictionary<string, IDictionary<Trackable, ISerializedAttributes>>? cache = null)
    {
        IDictionary<string, Trackable> children;
        if (save_type == SaveType.SAVEDMODEL)
        {
            Debug.Assert(cache is not null);
            children = TrackableSavedModelSaver.trackable_children(cache);
        }
        else
        {
            children = new Dictionary<string, Trackable>();
        }

        return children.Concat(base._trackable_children(save_type, cache)).GroupBy(x => x.Key).Select(g => g.First()).ToDictionary(x => x.Key, x => x.Value);
    }
}