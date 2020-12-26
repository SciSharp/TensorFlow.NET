using System;
using System.Collections.Generic;
using System.Text;
using Google.Protobuf;
using Google.Protobuf.Collections;

namespace Tensorflow.Contexts
{
    public class FunctionCallOptions
    {
        public string config_proto_serialized()
        {
            var config = new ConfigProto
            {
                AllowSoftPlacement = true,
            };
            return config.ToByteString().ToStringUtf8();
        }
    }
}
