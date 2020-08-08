using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class InitializerArgs
    {
        public string Name { get; set; }
        public TensorShape Shape { get; set; }
        public TF_DataType DType { get; set; }
        public bool? VerifyShape { get; set; } = null;

        public InitializerArgs(TensorShape shape, 
            TF_DataType dtype = TF_DataType.DtInvalid,
            bool? verify_shape = null,
            string name = null)
        {
            Shape = shape;
            DType = dtype;
            VerifyShape = verify_shape;
            Name = name;
        }
    }
}
