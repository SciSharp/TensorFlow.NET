/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using System;
using System.Diagnostics;
using System.Linq;
using Tensorflow.Eager;
using static Tensorflow.Binding;
using Google.Protobuf;
using Tensorflow.Device;
using System.Collections.Generic;

namespace Tensorflow.Contexts
{
    /// <summary>
    /// Environment in which eager operations execute.
    /// </summary>
    public sealed partial class Context
    {
        ContextDevicePlacementPolicy _device_policy;
        bool _log_device_placement;
        Dictionary<PhysicalDevice, bool> _memory_growth_map = new Dictionary<PhysicalDevice, bool>();

        public void log_device_placement(bool enable)
        {
            if (_handle != null)
                c_api.TFE_ContextSetLogDevicePlacement(_handle, enable, tf.Status.Handle);
            _log_device_placement = enable;
            // _thread_local_data.function_call_options = null;
        }

        public bool get_memory_growth(string device_type)
        {
            foreach(var map in _memory_growth_map)
            {
                if (map.Key.DeviceType == device_type)
                    return map.Value;
            }
            return false;
        }

        public void set_memory_growth(PhysicalDevice device, bool enable)
        {
            _memory_growth_map[device] = enable;
        }

        public PhysicalDevice[] list_physical_devices(string device_type = null)
        {
            using var opts = c_api.TFE_NewContextOptions();
            using var ctx = c_api.TFE_NewContext(opts, tf.Status.Handle);
            using var devices = c_api.TFE_ContextListDevices(ctx, tf.Status.Handle);
            tf.Status.Check(true);

            int num_devices = c_api.TF_DeviceListCount(devices);
            var results = new List<PhysicalDevice>();
            for (int i = 0; i < num_devices; ++i)
            {
                var dev_type = c_api.StringPiece(c_api.TF_DeviceListType(devices, i, tf.Status.Handle));
                tf.Status.Check(true);

                if (dev_type.StartsWith("XLA"))
                    continue;

                if (device_type == null || dev_type == device_type)
                {
                    var dev_name = c_api.TF_DeviceListName(devices, i, tf.Status.Handle);
                    tf.Status.Check(true);

                    results.Add(new PhysicalDevice
                    {
                        DeviceName = dev_name,
                        DeviceType = dev_type
                    });
                }
            }

            return results.ToArray();
        }
    }
}
