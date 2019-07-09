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
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations;

namespace Tensorflow
{
    public class control_flow_util
    {
        /// <summary>
        /// Return true if `op` is an Exit.
        /// </summary>
        /// <param name="op"></param>
        /// <returns></returns>
        public static bool IsLoopExit(Operation op)
        {
            return op.type == "Exit" || op.type == "RefExit";
        }

        /// <summary>
        /// Return true if `op` is a Switch.
        /// </summary>
        /// <param name="op"></param>
        /// <returns></returns>
        public static bool IsSwitch(Operation op)
        {
            return op.type == "Switch" || op.type == "RefSwitch";
        }

        /// <summary>
        /// Return the control flow context for the output of an op.
        /// </summary>
        public static ControlFlowContext GetOutputContext(Operation op)
        {
            var ctxt = op._get_control_flow_context();
            // Exit nodes usually have a control flow context, except in the case where the
            // exit node was imported via import_graph_def (in which case no nodes have
            // control flow contexts).
            if (ctxt != null && IsLoopExit(op))
                ctxt = ctxt.outer_context;
            return ctxt;
        }
    }
}
