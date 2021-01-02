using System;
using System.Collections.Generic;
using System.Text;
using Google.Protobuf;
using static Tensorflow.Binding;

namespace Tensorflow.Contexts
{
    public class FunctionCallOptions
    {
        public ConfigProto Config { get; set; }

        public string config_proto_serialized()
        {
            return Config.ToByteString().ToStringUtf8();
        }
    }
}
