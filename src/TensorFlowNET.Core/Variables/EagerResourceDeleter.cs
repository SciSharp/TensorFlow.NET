using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Variables
{
    public class EagerResourceDeleter : DisposableObject
    {
        Tensor _tensor;
        string _handle_device;
        public EagerResourceDeleter(Tensor handle, string handle_device)
        {
            _tensor = handle;
            _handle = handle.EagerTensorHandle.DangerousGetHandle();
            _handle_device = handle_device;
            
            bool success = false;
            handle.EagerTensorHandle.DangerousAddRef(ref success);
        }

        protected override void DisposeUnmanagedResources(IntPtr handle)
        {
            // gen_resource_variable_ops.destroy_resource_op(_tensor, ignore_lookup_error: true);

            tf.device(_handle_device);
            tf.Runner.TFE_Execute(tf.Context, _handle_device, "DestroyResourceOp",
                new[] { _tensor },
                new object[] { "ignore_lookup_error", true }, 0);
            
            _tensor.EagerTensorHandle.DangerousRelease();
        }
    }
}
