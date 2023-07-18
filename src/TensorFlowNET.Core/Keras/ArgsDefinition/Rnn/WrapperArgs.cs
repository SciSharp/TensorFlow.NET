using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;


namespace Tensorflow.Keras.ArgsDefinition
{
    public class WrapperArgs : AutoSerializeLayerArgs
    {
        [JsonProperty("layer")]
        public ILayer Layer { get; set; }

        public WrapperArgs(ILayer layer) 
        { 
            Layer = layer;
        }

        public static implicit operator WrapperArgs(BidirectionalArgs args)
            => new WrapperArgs(args.Layer);
    }

}
