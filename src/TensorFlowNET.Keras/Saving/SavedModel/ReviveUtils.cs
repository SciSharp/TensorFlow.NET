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

        public static void _revive_setter(object obj, object name, object value)
        {
            Debug.Assert(name is string);
            Debug.Assert(obj is Layer);
            Layer layer = (Layer)obj;
            if (KerasObjectLoader.PUBLIC_ATTRIBUTES.ContainsKey(name as string))
            {
                if (value is Trackable trackable)
                {
                    layer._track_trackable(trackable, name as string);
                }
                if (layer.SerializedAttributes is null)
                {
                    layer.SerializedAttributes = new Dictionary<string, object>();
                }
                layer.SerializedAttributes[name as string] = value;
            }
            else if (layer is Functional functional && Regex.Match(name as string, @"^layer(_with_weights)?-[\d+]").Success)
            {
                Debug.Assert(value is Trackable);
                functional._track_trackable(value as Trackable, name as string);
            }
            else
            {
                layer.SetAttr(name as string, value);
            }
        }
    }
}
