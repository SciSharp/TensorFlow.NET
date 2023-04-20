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

using Google.Protobuf;
using System;
using System.Diagnostics;
using System.Linq;
using Tensorflow.Common.Extensions;

namespace Tensorflow.Contexts
{
    /// <summary>
    /// Environment in which eager operations execute.
    /// </summary>
    public sealed partial class Context
    {
        protected Device.PhysicalDevice[] _physical_devices;
        protected Dictionary<Device.PhysicalDevice, int> _physical_device_to_index;
        ConfigProto _config;
        public ConfigProto Config
        {
            get
            {
                _initialize_physical_devices();

                var config = new ConfigProto();
                if(_config is not null)
                {
                    config.MergeFrom(_config);
                }
                config.LogDevicePlacement = _log_device_placement;

                config.DeviceCount["CPU"] = 0;
                config.DeviceCount["GPU"] = 0;
                foreach(var dev in _physical_devices)
                {
                    if (config.DeviceCount.ContainsKey(dev.DeviceType))
                    {
                        config.DeviceCount[dev.DeviceType] += 1;
                    }
                    else
                    {
                        config.DeviceCount[dev.DeviceType] = 1;
                    }
                }

                var gpu_options = _compute_gpu_options();
                config.GpuOptions = GPUOptions.Parser.ParseFrom(gpu_options.ToByteArray());

                return config;
            }
            set
            {
                _config = value;
            }
        }

        protected void _initialize_physical_devices(bool reinitialize = false)
        {
            if(!reinitialize && _physical_devices is not null)
            {
                return;
            }
            var devs = list_physical_devices();
            _physical_devices = devs.Select(d => new Device.PhysicalDevice()
            {
                DeviceName = d.DeviceName,
                DeviceType = d.DeviceType
            }).ToArray();
            _physical_device_to_index = _physical_devices.Select((p, i) => new KeyValuePair<Device.PhysicalDevice, int>(p, i))
                .ToDictionary(x => x.Key, x => x.Value);

            _import_config();
        }

        protected void _import_config()
        {
            if(_config is null)
            {
                return;
            }
            if(!_config.DeviceCount.TryGetValue("CPU", out var num_cpus))
            {
                num_cpus = 1;
            }
            if(num_cpus != 1)
            {
                // TODO(Rinne): implement it.
            }

            var gpus = _physical_devices.Where(d => d.DeviceType == "GPU");
            if(gpus.Count() == 0)
            {
                return;
            }

            if(!_config.DeviceCount.TryGetValue("GPU", out var gpu_count))
            {
                gpu_count = 0;
            }

            // TODO(Rinne): implement it.
        }

        ConfigProto MergeConfig()
        {
            Config.LogDevicePlacement = _log_device_placement;
            // var gpu_options = _compute_gpu_options();
            // Config.GpuOptions.AllowGrowth = gpu_options.AllowGrowth;
            return Config;
        }

        GPUOptions _compute_gpu_options()
        {
            // By default, TensorFlow maps nearly all of the GPU memory of all GPUs 
            // https://www.tensorflow.org/guide/gpu
            return new GPUOptions()
            {
                AllowGrowth = get_memory_growth("GPU")
            };
        }
    }
}
