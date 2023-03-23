using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Eager;

namespace Tensorflow.Operations
{
    public static class handle_data_util
    {
        public static void copy_handle_data(Tensor source_t, Tensor target_t)
        {
            if(target_t.dtype == dtypes.resource || target_t.dtype == dtypes.variant)
            {
                SafeTensorHandle handle_data;
                if(source_t is EagerTensor)
                {
                    handle_data = source_t.Handle;
                }
                else
                {
                    handle_data = ops.get_resource_handle_data(source_t);
                }
                throw new NotImplementedException();
                //if(handle_data is not null && handle_data.)
            }
        }
    }
}
