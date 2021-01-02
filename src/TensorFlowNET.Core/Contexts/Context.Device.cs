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

namespace Tensorflow.Contexts
{
    /// <summary>
    /// Environment in which eager operations execute.
    /// </summary>
    public sealed partial class Context
    {
        ContextDevicePlacementPolicy _device_policy;
        bool _log_device_placement;

        public void log_device_placement(bool enable)
        {
            if (_handle != null)
                c_api.TFE_ContextSetLogDevicePlacement(_handle, enable, tf.Status.Handle);
            _log_device_placement = enable;
            // _thread_local_data.function_call_options = null;
        }
    }
}
