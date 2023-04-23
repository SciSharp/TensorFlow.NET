using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Utils;
using Tensorflow.Keras.Saving.SavedModel;

namespace Tensorflow.Keras.Saving.SavedModel
{
    public class RevivedLayer: Layer
    {
        public static (RevivedLayer, Action<object, object, object>) init_from_metadata(KerasMetaData metadata)
        {
            LayerArgs args = new LayerArgs()
            {
                Name = metadata.Name,
                Trainable = metadata.Trainable
            };
            if(metadata.DType != TF_DataType.DtInvalid)
            {
                args.DType = metadata.DType;
            }
            if(metadata.BatchInputShape is not null)
            {
                args.BatchInputShape = metadata.BatchInputShape;
            }

            RevivedLayer revived_obj = new RevivedLayer(args);

            // TODO(Rinne): implement `expects_training_arg`.
            var config = metadata.Config;
            if (generic_utils.validate_config(config))
            {
                revived_obj._config = new RevivedConfig() { Config = config };
            }
            if(metadata.InputSpec is not null)
            {
                throw new NotImplementedException();
            }
            if(metadata.ActivityRegularizer is not null)
            {
                throw new NotImplementedException();
            }
            // TODO(Rinne): `_is_feature_layer`
            if(metadata.Stateful is not null)
            {
                revived_obj.stateful = metadata.Stateful.Value;
            }

            return (revived_obj, ReviveUtils._revive_setter);
        }

        protected RevivedConfig _config = null;

        public object keras_api
        {
            get
            {
                if (SerializedAttributes.TryGetValue(SavedModel.Constants.KERAS_ATTR, out var value))
                {
                    return value;
                }
                else
                {
                    return null;
                }
            }
        }

        protected RevivedLayer(LayerArgs args): base(args)
        {

        }

        public override string ToString()
        {
            return $"Customized keras layer: {Name}.";
        }

        public override IKerasConfig get_config()
        {
            return _config;
        }
    }
}
