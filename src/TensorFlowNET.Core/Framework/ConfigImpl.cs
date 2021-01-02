using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;
using Tensorflow.Device;

namespace Tensorflow.Framework
{
    public class ConfigImpl
    {
        /// <summary>
        /// Return a list of physical devices visible to the host runtime.
        /// </summary>
        /// <param name="device_type">CPU, GPU, TPU</param>
        /// <returns></returns>
        public PhysicalDevice[] list_physical_devices(string device_type = null)
            => tf.Context.list_physical_devices(device_type: device_type);

        public Experimental experimental => new Experimental();

        public class Experimental
        {
            public void set_memory_growth(PhysicalDevice device, bool enable)
                => tf.Context.set_memory_growth(device, enable);
        }
    }
}
