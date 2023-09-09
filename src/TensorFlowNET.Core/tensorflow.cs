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

using Razorvine.Pickle;
using Serilog;
using Serilog.Core;
using System.Reflection;
using System.Threading;
using Tensorflow.Contexts;
using Tensorflow.Eager;
using Tensorflow.Gradients;
using Tensorflow.Keras;
using Tensorflow.NumPy.Pickle;

namespace Tensorflow
{
    public delegate Tensor[] BackwardFunction(Tensor[] grads, long[] unneeded_gradients);

    public partial class tensorflow
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

        public OpDefLibrary OpDefLib;
        public Logger Logger;

        ThreadLocal<Status> _status = new ThreadLocal<Status>(() => new Status());
        public Status Status => _status.Value;

        ThreadLocal<Context> _context = new ThreadLocal<Context>(() => new Context());
        public Context Context => _context.Value;

        ThreadLocal<IEagerRunner> _runner = new ThreadLocal<IEagerRunner>(() => new EagerRunner());
        public IEagerRunner Runner => _runner.Value;

        private IKerasApi _keras;
        public IKerasApi keras 
        { 
            get
            {
                if (_keras != null)
                {
                    return _keras;
                }

                var k = Assembly.Load("Tensorflow.Keras");
                var cls = k.GetTypes().FirstOrDefault(x => x.GetInterfaces().Contains(typeof(IKerasApi)));
                if (cls != null)
                {
                    _keras = Activator.CreateInstance(cls) as IKerasApi;
                    return _keras;
                }
                else
                {
                    throw new Exception("Can't find keras library.");
                }
            }
        }

        public tensorflow()
        {
            Logger = new LoggerConfiguration()
                .MinimumLevel.Error()
                .WriteTo.Console()
                .CreateLogger();

            OpDefLib = new OpDefLibrary();
            InitGradientEnvironment();

            try
            {
                var handle = c_api.TF_Version();
            }
            catch (DllNotFoundException)
            {
                throw new RuntimeError("Tensorflow.NET cannot find a backend. Please install one of the following packages for your program: " +
                    "SciSharp.TensorFlow.Redist, SciSharp.TensorFlow.Redist-Linux-GPU, SciSharp.TensorFlow.Redist-Windows-GPU. For more details, " +
                    "please visit https://github.com/SciSharp/TensorFlow.NET. If it still not work after installing the backend, please submit an " +
                    "issue to https://github.com/SciSharp/TensorFlow.NET/issues");
            }

            // register numpy reconstructor for pickle
            Unpickler.registerConstructor("numpy.core.multiarray", "_reconstruct", new MultiArrayConstructor());
            Unpickler.registerConstructor("numpy", "dtype", new DtypeConstructor());
        }

        public string VERSION => c_api.StringPiece(c_api.TF_Version());

        private void InitGradientEnvironment()
        {
            _tapeSet = new GradientTape();
            ops.RegisterFromAssembly();
        }

        public ResourceVariable Variable<T>(T data,
            bool trainable = true,
            bool validate_shape = true,
            bool use_resource = true,
            string name = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            VariableAggregation aggregation = VariableAggregation.None,
            Shape shape = null)
            => new ResourceVariable(data,
                    trainable: trainable,
                    validate_shape: validate_shape,
                    name: name,
                    dtype: dtype,
                    aggregation: aggregation,
                    shape: shape);

        public Tensor placeholder(TF_DataType dtype, Shape shape = null, string name = null)
            => array_ops.placeholder(dtype, shape, name);

        public void enable_eager_execution()
            => Context.eager_mode();

        public Session get_default_session()
            => ops.get_default_session();

        public Session Session()
            => compat.v1.Session();

        public Session Session(Graph graph, ConfigProto config = null)
        {
            return new Session(graph, config: config).as_default();
        }

        public Session Session(ConfigProto config)
        {
            return new Session(null, config).as_default();
        }
    }
}
