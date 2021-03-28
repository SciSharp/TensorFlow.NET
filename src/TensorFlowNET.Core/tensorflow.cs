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

using System.Collections.Generic;
using Serilog;
using Serilog.Core;
using Tensorflow.Contexts;
using Tensorflow.Eager;
using Tensorflow.Gradients;

namespace Tensorflow
{
    public delegate Tensor[] BackwardFunction(Tensor[] grads, long[] unneeded_gradients);

    public partial class tensorflow : ITensorFlowObject
    {
        public TF_DataType byte8 = TF_DataType.TF_UINT8;
        public TF_DataType int8 = TF_DataType.TF_INT8;
        public TF_DataType int16 = TF_DataType.TF_INT16;
        public TF_DataType int32 = TF_DataType.TF_INT32;
        public TF_DataType int64 = TF_DataType.TF_INT64;
        public TF_DataType float16 = TF_DataType.TF_HALF;
        public TF_DataType float32 = TF_DataType.TF_FLOAT;
        public TF_DataType float64 = TF_DataType.TF_DOUBLE;
        public TF_DataType @bool = TF_DataType.TF_BOOL;
        public TF_DataType chars = TF_DataType.TF_STRING;
        public TF_DataType @string = TF_DataType.TF_STRING;

        public Status Status;
        public OpDefLibrary OpDefLib;
        public Context Context;
        public IEagerRunner Runner;
        public Logger Logger;

        public tensorflow()
        {
            Logger = new LoggerConfiguration()
                .MinimumLevel.Error()
                .WriteTo.Console()
                .CreateLogger();

            Status = new Status();
            Context = new Context();
            OpDefLib = new OpDefLibrary();
            ConstructThreadingObjects();
            InitGradientEnvironment();
            Runner = new EagerRunner();
        }

        public string VERSION => c_api.StringPiece(c_api.TF_Version());

        private void InitGradientEnvironment()
        {
            ops.RegisterFromAssembly();
        }

        public ResourceVariable Variable<T>(T data,
            bool trainable = true,
            bool validate_shape = true,
            bool use_resource = true,
            string name = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            VariableAggregation aggregation = VariableAggregation.None,
            int[] shape = null)
            => new ResourceVariable(data,
                    trainable: trainable,
                    validate_shape: validate_shape,
                    name: name,
                    dtype: dtype,
                    aggregation: aggregation,
                    shape: shape);

        public Tensor placeholder(TF_DataType dtype, TensorShape shape = null, string name = null)
            => array_ops.placeholder(dtype, shape, name);

        public void enable_eager_execution()
            => Context.eager_mode();

        public Session get_default_session()
            => ops.get_default_session();

        public Session Session()
        {
            return new Session().as_default();
        }

        public Session Session(Graph graph, ConfigProto config = null)
        {
            return new Session(graph, config: config).as_default();
        }

        public Session Session(ConfigProto config)
        {
            return new Session(null, config).as_default();
        }

        List<ITape> tape_set;
        public List<ITape> GetTapeSet()
        {
            if (tape_set == null)
            {
                tape_set = new List<ITape>();
            }

            return tape_set;
        }

        public void __init__()
        {

        }

        public void __enter__()
        {

        }

        public void __exit__()
        {

        }

        public void __del__()
        {

        }

        public void Dispose()
        {

        }
    }
}
