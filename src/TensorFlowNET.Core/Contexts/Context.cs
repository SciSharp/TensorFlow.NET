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
using Tensorflow.Eager;

namespace Tensorflow.Contexts
{
    /// <summary>
    /// Environment in which eager operations execute.
    /// </summary>
    public sealed class Context : IDisposable
    {
        public const int GRAPH_MODE = 0;
        public const int EAGER_MODE = 1;

        int defaultExecutionMode = EAGER_MODE;
        public string DeviceName { get; set; } = "";
        public string ScopeName { get; set; } = "";
        bool initialized = false;
        bool isEager;
        ContextSwitchStack contextSwitches;

        public SafeContextHandle Handle { get; }

        public Context(ContextOptions opts, Status status)
        {
            Handle = c_api.TFE_NewContext(opts.Handle, status.Handle);
            status.Check(true);
            isEager = defaultExecutionMode == EAGER_MODE;
            contextSwitches = new ContextSwitchStack(isEager);
            initialized = true;
        }

        /// <summary>
        /// Initialize handle and devices if not already done so.
        /// </summary>
        public void ensure_initialized()
        {
            if (initialized)
                return;
            initialized = true;
        }

        public void start_step()
            => c_api.TFE_ContextStartStep(Handle);

        public void end_step()
            => c_api.TFE_ContextEndStep(Handle);

        /// <summary>
        /// Checks whether the current thread has eager execution enabled.
        /// </summary>
        /// <returns></returns>
        public bool executing_eagerly()
            => isEager;

        public string shared_name(string name = null)
            => !string.IsNullOrEmpty(name) || !executing_eagerly() ? 
                name : 
                "cd2c89b7-88b7-44c8-ad83-06c2a9158347";

        public void graph_mode()
            => mode(false);

        public void eager_mode()
            => mode(true);

        void mode(bool mode)
        {
            isEager = mode;
        }

        public void Dispose()
            => Handle.Dispose();
    }
}
