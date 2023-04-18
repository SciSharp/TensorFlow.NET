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
using Tensorflow.Exceptions;
using System.Collections.Generic;

namespace Tensorflow.Contexts
{
    /// <summary>
    /// Environment in which eager operations execute.
    /// </summary>
    public sealed partial class Context
    {
        internal static Dictionary<(string, string), (string, DeviceSpec)> _device_parsing_cache = new();
        internal List<LogicalDevice> _logical_devices = null;
        internal List<string> _context_devices = null;
        
        ContextDevicePlacementPolicy _device_policy;
        bool _log_device_placement;
        int _num_gpus;
        Dictionary<PhysicalDevice, bool> _memory_growth_map = new Dictionary<PhysicalDevice, bool>();

        public string DeviceName { get; set; } = "";
        public DeviceSpec DeviceSpec { get; set; } = null;

        internal List<string> Devices
        {
            get
            {
                if(_context_devices is null)
                {
                    throw new AssertionError("Context must be initialized first.");
                }
                return _context_devices;
            }
        }

        public void log_device_placement(bool enable)
        {
            if (_handle != null)
                c_api.TFE_ContextSetLogDevicePlacement(_handle, enable, tf.Status);
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
            using var ctx = c_api.TFE_NewContext(opts, tf.Status);
            using var devices = c_api.TFE_ContextListDevices(ctx, tf.Status);
            tf.Status.Check(true);

            int num_devices = c_api.TF_DeviceListCount(devices);
            var results = new List<PhysicalDevice>();
            for (int i = 0; i < num_devices; ++i)
            {
                var dev_type = c_api.StringPiece(c_api.TF_DeviceListType(devices, i, tf.Status));
                tf.Status.Check(true);

                if (dev_type.StartsWith("XLA"))
                    continue;

                if (device_type == null || dev_type == device_type)
                {
                    var dev_name = c_api.TF_DeviceListName(devices, i, tf.Status);
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

        public bool is_custom_device(string device_name)
        {
            return false;
            // TODO(Rinne): After tf2.11 TFE_IsCustomDevice has been added to C APIs.
            //ensure_initialized();
            //return c_api.TFE_IsCustomDevice(_handle, device_name);
        }

        public EagerDeviceContext device(string name)
        {
            return new EagerDeviceContext(this, name);
        }

        internal void _set_device(string device_name, DeviceSpec device_spec)
        {
            DeviceSpec = device_spec;
            DeviceName = device_name;
        }

        internal void _initialize_logical_devices()
        {
            List<LogicalDevice> logical_devices = new();
            List<string> context_devices = new();
            Status status = new();
            var device_list = c_api.TFE_ContextListDevices(_handle, status);
            status.Check(true);
            try
            {
                this._num_gpus = 0;
                string current_job = null;
                int current_task = -1;
                for(int i = 0; i < c_api.TF_DeviceListCount(device_list); i++)
                {
                    var dev_name = c_api.TF_DeviceListName(device_list, i, status);
                    status.Check(true);
                    context_devices.Add(DeviceUtils.canonical_name(dev_name));
                    var spec = DeviceSpec.from_string(dev_name);
                    if(spec.Job == "localhost")
                    {
                        spec = spec.replace(job: null, replica: -1, task: -1);
                    }
                    logical_devices.Add(new LogicalDevice(spec.ToString(), spec.DeviceType));
                    var dev_type_memory = c_api.TF_DeviceListType(device_list, i, status);
                    var dev_type = c_api.StringPiece(dev_type_memory);
                    status.Check(true);
                    if(dev_type == "GPU" && spec.Job == current_job && spec.Task == current_task)
                    {
                        _num_gpus++;
                    }
                }
            }
            finally
            {
                _logical_devices = logical_devices;
                _context_devices = context_devices;
            }
        }
    }

    public record class LogicalDevice(string name, string device_type);
}
