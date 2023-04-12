using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Device
{
    internal static class DeviceUtils
    {
        public static string canonical_name(string device)
        {
            if(device is null)
            {
                return "";
            }
            return DeviceSpec.from_string(device).ToString();
        }
        public static string canonical_name(DeviceSpec device)
        {
            if (device is null)
            {
                return "";
            }
            return device.ToString();
        }
    }
}
