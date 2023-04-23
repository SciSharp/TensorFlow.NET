using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Utils;

namespace Tensorflow.Keras.Saving.SavedModel
{
    public class RevivedNetwork: RevivedLayer
    {
        private RevivedNetwork(LayerArgs args) : base(args)
        {
            
        }

        public static (RevivedNetwork, Action<object, object, object>) init_from_metadata(KerasMetaData metadata)
        {
            RevivedNetwork revived_obj = new(new LayerArgs() { Name = metadata.Name });

            // TODO(Rinne): with utils.no_automatic_dependency_tracking_scope(revived_obj)
            // TODO(Rinne): revived_obj._expects_training_arg
            var config = metadata.Config;
            if (generic_utils.validate_config(config))
            {
                revived_obj._config = new RevivedConfig() { Config = config };
            }
            if(metadata.ActivityRegularizer is not null)
            {
                throw new NotImplementedException();
            }

            return (revived_obj, ReviveUtils._revive_setter);
        }

        public override string ToString()
        {
            return $"Customized keras Network: {Name}.";
        }
    }
}
