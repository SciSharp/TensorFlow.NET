using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Device;

namespace Tensorflow.Contexts
{
    public class EagerDeviceContext : ITensorFlowObject
    {
        private Context _ctx;
        private string _device_name;
        private Stack<(string, DeviceSpec, DeviceSpec)> _stack;

        public EagerDeviceContext(Context ctx, string device_name)
        {
            _ctx = ctx;
            _device_name = device_name;
            _stack = new Stack<(string, DeviceSpec, DeviceSpec)>();
        }
        public void __enter__()
        {
            var ctx = _ctx;
            var old_device_name = ctx.DeviceName;
            var old_device_spec = ctx.DeviceSpec; 
            var new_device_name = _device_name;
            var cache_key = (old_device_name, new_device_name);
            DeviceSpec new_device_spec;
            if (Context._device_parsing_cache.ContainsKey(cache_key))
            {
                (new_device_name, new_device_spec) = Context._device_parsing_cache[cache_key];
            }
            else
            {
                if(new_device_name is not null)
                {
                    var device_spec = DeviceSpec.from_string(new_device_name);
                    if (!string.IsNullOrEmpty(old_device_name))
                    {
                        new_device_spec = new DeviceSpec(old_device_spec);
                    }
                    else
                    {
                        ctx.ensure_initialized();
                        new_device_spec = DeviceSpec.from_string(ctx._context_devices[0]);
                    }
                    new_device_spec = new_device_spec.make_merged_spec(device_spec);
                }
                else
                {
                    new_device_spec = DeviceSpec.from_string(ctx._context_devices[0]);
                }
                new_device_name = new_device_spec.ToString();
                Context._device_parsing_cache[cache_key] = (new_device_name, new_device_spec);
            }
            ctx._set_device(new_device_name, new_device_spec);
            _stack.Push((old_device_name, old_device_spec, new_device_spec));
        }

        public void __exit__()
        {
            var ctx = _ctx;
            var (old_device_name, old_device_spec, new_device_spec) = _stack.Pop();
            ctx._set_device(old_device_name, old_device_spec);
        }

        public void Dispose()
        {

        }
    }
}
