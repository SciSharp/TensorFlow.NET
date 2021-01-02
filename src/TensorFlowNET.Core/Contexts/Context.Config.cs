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

namespace Tensorflow.Contexts
{
    /// <summary>
    /// Environment in which eager operations execute.
    /// </summary>
    public sealed partial class Context
    {
        public ConfigProto Config { get; set; } = new ConfigProto
        {
            GpuOptions = new GPUOptions
            {
            }
        };

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
