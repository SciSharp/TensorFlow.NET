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
using Tensorflow.Util;

namespace Tensorflow.Contexts
{
    /// <summary>
    /// Environment in which eager operations execute.
    /// </summary>
    public sealed partial class Context : IDisposable
    {
        public const int GRAPH_MODE = 0;
        public const int EAGER_MODE = 1;

        int defaultExecutionMode = EAGER_MODE;
        public string DeviceName { get; set; } = "";
        public string ScopeName { get; set; } = "";
        bool initialized = false;
        ContextSwitchStack context_switches;
        public FunctionCallOptions FunctionCallOptions { get; }

        SafeContextHandle _handle;
        public SafeContextHandle Handle => _handle;

        int? _seed;
        Random _rng;

        public Context()
        {
            _device_policy = ContextDevicePlacementPolicy.DEVICE_PLACEMENT_SILENT;
            context_switches = new ContextSwitchStack(defaultExecutionMode == EAGER_MODE, false);
            initialized = false;
            FunctionCallOptions = new FunctionCallOptions();
        }

        /// <summary>
        /// Initialize handle and devices if not already done so.
        /// </summary>
        public void ensure_initialized()
        {
            if (initialized)
                return;

            Config = MergeConfig();
            FunctionCallOptions.Config = Config;
            var config_str = Config.ToByteArray();
            using var opts = new ContextOptions();
            using var status = new Status();
            c_api.TFE_ContextOptionsSetConfig(opts.Handle, config_str, (ulong)config_str.Length, status.Handle);
            status.Check(true);
            c_api.TFE_ContextOptionsSetDevicePlacementPolicy(opts.Handle, _device_policy);
            _handle = c_api.TFE_NewContext(opts.Handle, status.Handle);
            status.Check(true);
            initialized = true;
        }

        public void set_global_seed(int? seed)
        {
            _seed = seed;
            if (seed.HasValue)
                _rng = new Random(seed.Value);
            else
                _rng = null;
            // Also clear the kernel cache, to reset any existing seeds
            if (_handle != null)
                c_api.TFE_ContextClearCaches(_handle);
        }

        public int? global_seed()
            => _seed;

        public int? internal_operation_seed()
            => _rng?.Next(0, int.MaxValue);

        public void start_step()
            => c_api.TFE_ContextStartStep(_handle);

        public void end_step()
            => c_api.TFE_ContextEndStep(_handle);

        /// <summary>
        /// Checks whether the current thread has eager execution enabled.
        /// </summary>
        /// <returns></returns>
        [DebuggerStepThrough]
        public bool executing_eagerly()
        {
            if(context_switches.Count() == 0)
                tf.enable_eager_execution();

            return context_switches.Current().EagerMode;
        }

        public bool is_build_function()
            => context_switches.Current().IsBuildingFunction;

        public string shared_name(string name = null)
            => !string.IsNullOrEmpty(name) || !executing_eagerly() ?
                name :
                "cd2c89b7-88b7-44c8-ad83-06c2a9158347";

        public void graph_mode(bool isFunc = false)
            => context_switches.Push(false, isFunc);

        public void eager_mode(bool isFunc = false)
            => context_switches.Push(true, isFunc);

        public bool switched_to_graph(params object[] args)
        {
            var switching_to_graph = has_graph_arg(args) && tf.Context.executing_eagerly();
            if (switching_to_graph)
                tf.Context.graph_mode(tf.Context.is_build_function());
            return switching_to_graph;
        }

        public bool has_graph_arg(params object[] args)
        {
            var flatten_args = nest.flatten<object>(args);
            /*if (flatten_args.Count(x => x.GetType().IsValueType) == flatten_args.Count())
                return tf.Context.executing_eagerly() == false*/

            bool has_graph_arg = !tf.Context.executing_eagerly();
            foreach (var el in flatten_args)
            {
                if (el is Tensor tensor && !tensor.IsEagerTensor)
                {
                    has_graph_arg = true;
                    break;
                }
            }
            return has_graph_arg;
        }

        public void restore_mode()
        {
            context_switches.Pop();
            tf.get_default_graph();
        }

        public void reset_context()
        {
            ops.reset_uid();
            // tf.defaultSession = null;
            ops.reset_default_graph();
            context_switches.Clear();
            tf.Context.ensure_initialized();

            if (_handle != null)
                c_api.TFE_ContextClearCaches(_handle);
        }

        public void Dispose()
            => _handle.Dispose();
    }
}
