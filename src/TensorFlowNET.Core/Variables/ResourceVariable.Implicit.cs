using System;
using Tensorflow.Eager;

namespace Tensorflow
{
    public partial class ResourceVariable
    {
        public static implicit operator _VariableScopeStore(ResourceVariable variable)
        {
            return null;
        }

        public static implicit operator ResourceVariable(_VariableScopeStore store)
        {
            return null;
        }

        public static implicit operator Tensor(ResourceVariable var)
            => var._dense_var_to_tensor();

        public static implicit operator EagerTensor(ResourceVariable var)
            => var._dense_var_to_tensor() as EagerTensor;

        public static implicit operator IntPtr(ResourceVariable var)
            => var._handle;

        Tensor _dense_var_to_tensor(TF_DataType dtype = TF_DataType.DtInvalid,
            string name = null,
            bool as_ref = false)
        {
            return value();
        }

        public Tensor _TensorConversionFunction(TF_DataType dtype = TF_DataType.DtInvalid, string name = null, bool as_ref = false)
        {
            if (as_ref)
                return handle;
            else
                return GraphElement ?? read_value();
        }
    }
}
