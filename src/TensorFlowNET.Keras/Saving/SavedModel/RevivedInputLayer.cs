using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;

namespace Tensorflow.Keras.Saving.SavedModel
{
    public class RevivedInputLayer: InputLayer
    {
        protected RevivedConfig _config = null;
        private RevivedInputLayer(InputLayerArgs args): base(args)
        {
            
        }

        public override IKerasConfig get_config()
        {
            return _config;
        }

        public static (RevivedInputLayer, Action<object, object, object>) init_from_metadata(KerasMetaData metadata)
        {
            InputLayerArgs args = new InputLayerArgs()
            {
                Name = metadata.Name,
                DType = metadata.DType,
                Sparse = metadata.Sparse,
                Ragged = metadata.Ragged,
                BatchInputShape = metadata.BatchInputShape
            };

            RevivedInputLayer revived_obj = new RevivedInputLayer(args);

            revived_obj._config = new RevivedConfig() { Config = metadata.Config };

            return (revived_obj, Loader.setattr);
        }

        public override string ToString()
        {
            return $"Customized keras input layer: {Name}.";
        }
    }
}
