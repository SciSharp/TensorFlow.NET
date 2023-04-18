using System;
using System.Collections.Generic;
using System.Text;
using Google.Protobuf;
using Protobuf.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Contexts
{
    public class FunctionCallOptions
    {
        public ConfigProto Config { get; set; }
        public string ExecutorType { get; set; }

        public ByteString config_proto_serialized()
        {
            return Config.ToByteString();
        }
    }
}
