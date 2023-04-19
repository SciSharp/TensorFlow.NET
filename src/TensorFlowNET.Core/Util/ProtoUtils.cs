using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Util
{
    internal static class ProtoUtils
    {
        public static object GetSingleAttrValue(AttrValue value, AttrValue.ValueOneofCase valueCase)
        {
            return valueCase switch
            {
                AttrValue.ValueOneofCase.S => value.S.ToStringUtf8(),
                AttrValue.ValueOneofCase.I => value.I,
                AttrValue.ValueOneofCase.F => value.F,
                AttrValue.ValueOneofCase.B => value.B,
                AttrValue.ValueOneofCase.Type => value.Type,
                AttrValue.ValueOneofCase.Shape => value.Shape,
                AttrValue.ValueOneofCase.Tensor => value.Tensor,
                AttrValue.ValueOneofCase.Func => value.Func,
            };
        }
    }
}
