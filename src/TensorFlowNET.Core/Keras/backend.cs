using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras
{
    public class backend
    {
        public static void track_variable(RefVariable v)
        {

        }

        public static Tensor placeholder(int[] shape = null, 
            int ndim = -1, 
            TF_DataType dtype = TF_DataType.DtInvalid, 
            bool sparse = false, 
            string name = null)
        {
            if(sparse)
            {
                throw new NotImplementedException("placeholder sparse is true");
            }
            else
            {
                return gen_array_ops.placeholder(dtype: dtype, shape: new TensorShape(shape), name: name);
            }
        }

        public static Graph get_graph()
        {
            return ops.get_default_graph();
        }
    }
}
