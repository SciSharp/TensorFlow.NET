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

using Tensorflow.Operations;

namespace Tensorflow
{
    public partial class Operation
    {
        private ControlFlowContext _control_flow_context;

        /// <summary>
        /// Add this op to its control flow context.
        /// 
        /// This may add new ops and change this op's inputs. self.inputs must be
        /// available before calling this method.
        /// </summary>
        public void _control_flow_post_processing()
        {
            foreach (Tensor input_tensor in inputs)
                control_flow_util.CheckInputFromValidContext(this, input_tensor.op);

            if (_control_flow_context != null)
                _control_flow_context.AddOp(this);
        }

        public void _add_control_input(Operation op)
        {
            // c_api.TF_AddControlInput(_opDesc, op);
            //c_api.AddControlInput(graph, _handle, op);
        }

        public void _add_control_inputs(Operation[] ops)
        {
            foreach (var op in ops)
                _add_control_input(op);
        }

        public void _set_control_flow_context(ControlFlowContext ctx)
        {
            _control_flow_context = ctx;
        }

        public ControlFlowContext _get_control_flow_context()
        {
            return _control_flow_context;
        }

        public WhileContext GetWhileContext()
        {
            return _control_flow_context as WhileContext;
        }
    }
}
