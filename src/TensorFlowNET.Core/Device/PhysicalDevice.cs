using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Device
{
    public class PhysicalDevice
    {
        public string DeviceName { get; set; }
        public string DeviceType { get; set; }

        public override string ToString()
            => $"{DeviceType}: {DeviceName}";
    }
}
