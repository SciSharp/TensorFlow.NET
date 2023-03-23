using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Text.RegularExpressions;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Utils;
using Tensorflow.Train;

namespace Tensorflow.Keras.Saving.SavedModel
{
    internal static class ReviveUtils
    {
        public static T recursively_deserialize_keras_object<T>(JToken config)
        {
            throw new NotImplementedException();
            if(config is JObject jobject)
            {
                if (jobject.ContainsKey("class_name"))
                {
                    
                }
            }
        }

        public static void _revive_setter(object layer, object name, object value)
        {
            Debug.Assert(name is string);
            Debug.Assert(layer is Layer);
            if (KerasObjectLoader.PUBLIC_ATTRIBUTES.ContainsKey(name as string))
            {
                if (value is Trackable trackable)
                {
                    (layer as Layer)._track_trackable(trackable, name as string);
                }
                (layer as Layer).SerializedAttributes[name] = JToken.FromObject(value);
            }
            else if (layer is Functional functional && Regex.Match(name as string, @"^layer(_with_weights)?-[\d+]").Success)
            {
                Debug.Assert(value is Trackable);
                functional._track_trackable(value as Trackable, name as string);
            }
            else
            {
                var properties = layer.GetType().GetProperties();
                foreach (var p in properties)
                {
                    if ((string)name == p.Name)
                    {
                        if(p.GetValue(layer) is not null)
                        {
                            return;
                        }
                        p.SetValue(layer, value);
                        return;
                    }
                }
            }
        }
    }
}
