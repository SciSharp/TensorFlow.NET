using System.Collections.Generic;

namespace Tensorflow.Keras.Optimizers
{
    public class DeviceDType : IEqualityComparer<DeviceDType>
    {
        public string Device { get; set; }
        public TF_DataType DType { get; set; }

        public bool Equals(DeviceDType x, DeviceDType y)
        {
            return x.ToString() == y.ToString();
        }

        public int GetHashCode(DeviceDType obj)
        {
            return 0;
        }

        public override string ToString()
            => $"{Device}, {DType}";
    }
}
