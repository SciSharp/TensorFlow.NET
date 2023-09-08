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

using Tensorflow.NumPy;
using System;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using Tensorflow.Eager;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;

namespace Tensorflow
{
    /// <summary>
    /// A tensor is a generalization of vectors and matrices to potentially higher dimensions. 
    /// Internally, TensorFlow represents tensors as n-dimensional arrays of base datatypes.
    /// </summary>
    [SuppressMessage("ReSharper", "ConvertToAutoProperty")]
    public partial class Tensor : DisposableObject,
        ITensorOrOperation,
        ITensorOrTensorArray,
        IPackable<Tensor>,
        ICanBeFlattened
    {
        protected long _id;
        private readonly Operation _op;
        private readonly int _value_index;
        private TF_Output? _tf_output;
        private readonly TF_DataType _override_dtype;
        public long Id => _id;

        /// <summary>
        ///     The Graph that contains this tensor.
        /// </summary>
        public Graph graph => op?.graph;

        /// <summary>
        ///     The Operation that produces this tensor as an output.
        /// </summary>
        public Operation op => _op;
        public Tensor[] outputs => op?.outputs;

        /// <summary>
        /// The string name of this tensor.<br/>
        /// Tensor.name is meaningless when eager execution is enabled.
        /// </summary>
        public virtual string name => $"{(op == null ? "<unnamed>" : $"{op.name}:{_value_index}")}";

        /// <summary>
        ///     The index of this tensor in the outputs of its Operation.
        /// </summary>
        public int value_index => _value_index;

        /// <summary>
        ///     The DType of elements in this tensor.
        /// </summary>
        public virtual TF_DataType dtype => _handle == null ? _override_dtype : c_api.TF_TensorType(_handle);
        public virtual ulong bytesize => _handle == null ? 0 : c_api.TF_TensorByteSize(_handle);
        public ulong dtypesize => (ulong)dtype.get_datatype_size();
        public ulong size => _handle == null ? 0 : bytesize / dtypesize;
        public virtual IntPtr buffer => _handle == null ? IntPtr.Zero : c_api.TF_TensorData(_handle);
        public int num_consumers(TF_Output oper_out) => _handle == null ? 0 : c_api.TF_OperationOutputNumConsumers(oper_out);
        public int ndim => rank;

        /// <summary>
        ///     The name of the device on which this tensor will be produced, or null.
        /// </summary>
        public virtual string Device => op?.Device;
        public long[] dims => shape.dims;

        /// <summary>
        ///     Used for keep other pointer when do implicit operating
        /// </summary>
        public object Tag { get; set; }
        protected new SafeTensorHandle _handle;
        public virtual SafeTensorHandle Handle => _handle;
        public Tensorflow.CppShapeInferenceResult.Types.HandleData HandleData { get; internal set; }

        protected SafeEagerTensorHandle _eagerTensorHandle;
        /// <summary>
        /// TFE_TensorHandle
        /// </summary>
        public SafeEagerTensorHandle EagerTensorHandle => _eagerTensorHandle;

        /// <summary>
        ///     Returns the shape of a tensor.
        /// </summary>
        /// <remarks>https://www.tensorflow.org/api_docs/python/tf/shape</remarks>
        public Shape shape
        {
            get
            {
                if (rank < 0)
                    return Shape.Null;

                return GetShapeInternal();
            }

            set
            {
                SetShapeInternal(value);
                tf.Status.Check(true);
            }
        }

        protected virtual Shape GetShapeInternal()
        {
            var dims = new Shape(new long[rank]);

            if (_handle == null)
            {
                c_api.TF_GraphGetTensorShape(op.graph, _as_tf_output(), dims, rank, tf.Status);
            }
            else
            {
                for (int i = 0; i < rank; i++)
                    dims[i] = c_api.TF_Dim(_handle, i);
            }

            return dims;
        }

        protected virtual void SetShapeInternal(Shape value)
        {
            if (value is null || value.ndim == 0 || value.ndim == -1)
                c_api.TF_GraphSetTensorShape(op.graph.c_graph, _as_tf_output(), null, -1, tf.Status);
            else
                c_api.TF_GraphSetTensorShape(op.graph.c_graph, _as_tf_output(), value.dims, value.ndim, tf.Status);
        }

        public int[] _shape_tuple()
        {
            return rank < 0 ? null : shape.dims.Select(x => (int)x).ToArray();
        }

        /// <summary>
        ///     Updates the shape of this tensor.
        /// </summary>
        public void set_shape(Tensor shape)
        {
            // ReSharper disable once MergeConditionalExpression
            this.shape = shape is null ? null : shape.shape;
        }

        /// <summary>
        /// number of dimensions <br></br>
        /// -1 Unknown  <br></br>
        /// 0	Scalar (magnitude only) <br></br>
        /// 1	Vector (magnitude and direction) <br></br>
        /// 2	Matrix (table of numbers) <br></br>
        /// 3	3-Tensor (cube of numbers) <br></br>
        /// n	n-Tensor (you get the idea)
        /// </summary>
        /// <remarks>https://www.tensorflow.org/api_docs/python/tf/rank</remarks>
        public virtual int rank
        {
            get
            {
                if (_handle == null)
                {
                    var output = _as_tf_output();
                    Status status = new();
                    int ndim = c_api.TF_GraphGetTensorNumDims(op.graph, output, status);
                    status.Check(true);
                    return ndim;
                }

                return c_api.TF_NumDims(_handle);
            }
        }

        /// <summary>
        ///     Returns a list of Operations that consume this tensor.
        /// </summary>
        /// <returns></returns>
        public Operation[] consumers()
        {
            var output = _as_tf_output();
            var consumer_names = c_api.TF_OperationOutputConsumers_wrapper(output);
            return consumer_names.Select(x => graph.OperationByName(x)).ToArray();
        }

        public TF_Output _as_tf_output()
        {
            if (!_tf_output.HasValue)
                _tf_output = new TF_Output(op, _value_index);

            return _tf_output.Value;
        }
        
        public Tensor MaybeMove()
        {
            var tensor = c_api.TF_TensorMaybeMove(_handle);
            return tensor;
        }

        /// <summary>
        ///     Evaluates this tensor in a `Session`.
        /// </summary>
        /// <param name="feed_dict">A dictionary that maps `Tensor` objects to feed values.</param>
        /// <returns>A <see cref="NumPy"/> array corresponding to the value of this tensor.</returns>
        public NDArray eval(params FeedItem[] feed_dict)
        {
            return ops._eval_using_default_session(this, feed_dict, graph);
        }

        /// <summary>
        ///     Evaluates this tensor in a `Session`.
        /// </summary>
        /// <param name="feed_dict">A dictionary that maps `Tensor` objects to feed values.</param>
        /// <param name="session">The `Session` to be used to evaluate this tensor.</param>
        /// <returns>A <see cref="NumPy"/> array corresponding to the value of this tensor.</returns>
        public NDArray eval(Session session, params FeedItem[] feed_dict)
        {
            return ops._eval_using_default_session(this, feed_dict, graph, session);
        }

        public override string ToString()
        {
            // this can throw IndexOutOfRangeException 
            switch (rank)
            {
                case -1:
                    return $"tf.Tensor '{name}' shape={shape} dtype={dtype.as_numpy_name()}";
                case 0:
                    return $"tf.Tensor '{name}' shape={shape} dtype={dtype.as_numpy_name()}";
                default:
                    return $"tf.Tensor '{name}' shape={shape} dtype={dtype.as_numpy_name()}";
            }
        }

        protected override void DisposeUnmanagedResources(IntPtr handle)
        {

        }

        public bool IsDisposed => _disposed;
    }
}