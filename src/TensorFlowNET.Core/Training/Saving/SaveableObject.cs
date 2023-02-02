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

using Tensorflow.Checkpoint;

namespace Tensorflow
{
    public class MySaveableObject
    {
        protected Maybe<Tensor, BaseResourceVariable> _op;
        public Tensor op
        {
            get
            {
                if(_op.DataType == typeof(Tensor))
                {
                    return _op.GetValueA();
                }
                else
                {
                    throw new TypeError("The _op is not a tensor.");
                }
            }
            set
            {
                _op = value;
            }
        }
        public SaveSpec[] specs;
        public string name;
        public string device;

        public MySaveableObject()
        {

        }

        public MySaveableObject(Tensor var, string slice_spec, string name)
        {

        }

        public MySaveableObject(Tensor op, SaveSpec[] specs, string name)
        {
            this._op = op;
            this.specs = specs;
            this.name = name;
        }

        public virtual Operation restore(Tensor[] restored_tensors, Shape[] restored_shapes = null)
        {
            var restored_tensor = restored_tensors[0];
            return gen_state_ops.assign(op,
                restored_tensor,
                validate_shape: restored_shapes == null && op.shape.IsFullyDefined);
        }
    }

    public class NoRestoreSaveable: MySaveableObject
    {
        public NoRestoreSaveable(Tensor tensor, string name, TF_DataType dtype = TF_DataType.DtInvalid, string? device = null) : base(tensor,
            new SaveSpec[] { new SaveSpec(tensor, "", name, dtype) }, name)
        {
            
        }

        public override Operation restore(Tensor[] restored_tensors, Shape[] restored_shapes = null)
        {
            return control_flow_ops.no_op();
        }
    }
}
