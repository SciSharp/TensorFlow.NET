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
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Eager;
using Tensorflow.Graphs;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public static class tensor_util
    {
        /// <summary>
        /// Returns the constant value of the given tensor, if efficiently calculable.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="partial"></param>
        /// <returns></returns>
        public static NDArray constant_value(Tensor tensor, bool partial = false)
        {
            if (tensor is NDArray nd)
                return nd;
            else if (tensor is EagerTensor)
                return tensor.numpy();

            NDArray ret = _ConstantValue(tensor, partial);
            if (!(ret is null))
                tensor.graph.prevent_feeding(tensor);

            return ret;
        }

        private static NDArray _ConstantValue(Tensor tensor, bool partial)
        {
            switch (tensor.op.type)
            {
                case "Const":
                    return MakeNdarray(tensor.op.get_attr("value") as TensorProto);
                default:
                    return null;
            }
        }

        public static NDArray MakeNdarray(TensorProto tensor)
        {
            var shape = new Shape(tensor.TensorShape.Dim.Select(x => x.Size).ToArray());
            var num_elements = shape.size;
            var tensor_dtype = tensor.Dtype.as_tf_dtype();

            if (shape.ndim > 0 && tensor.TensorContent.Length > 0)
            {
                return np.frombuffer(tensor.TensorContent.ToByteArray(), shape, tensor_dtype);
            }
            else if (tensor.Dtype == DataType.DtHalf || tensor.Dtype == DataType.DtBfloat16)
            {
                return np.array(tensor.HalfVal.ToArray()).reshape(shape);
            }
            else if (tensor.Dtype == DataType.DtFloat)
            {
                return np.array(tensor.FloatVal.ToArray()).reshape(shape);
            }
            else if (new DataType[] { DataType.DtInt32, DataType.DtUint8 }.Contains(tensor.Dtype))
            {
                return np.array(tensor.IntVal.ToArray()).reshape(shape);
            }
            else if (tensor.Dtype == DataType.DtBool)
            {
                return np.array(tensor.BoolVal.ToArray()).reshape(shape);
            }

            throw new NotImplementedException("MakeNdarray");
        }

        private static readonly TF_DataType[] quantized_types = new TF_DataType[]
        {
            TF_DataType.TF_QINT8, TF_DataType.TF_QUINT8, TF_DataType.TF_QINT16, TF_DataType.TF_QUINT16,
            TF_DataType.TF_QINT32
        };

        /// <summary>
        /// Create a TensorProto, invoked in graph mode
        /// </summary>
        /// <param name="values"></param>
        /// <param name="dtype"></param>
        /// <param name="shape"></param>
        /// <param name="verify_shape"></param>
        /// <param name="allow_broadcast"></param>
        /// <returns></returns>
        public static TensorProto make_tensor_proto(object values, TF_DataType dtype = TF_DataType.DtInvalid, Shape? shape = null, bool verify_shape = false, bool allow_broadcast = false)
        {
            if (allow_broadcast && verify_shape)
                throw new ValueError("allow_broadcast and verify_shape are not both allowed.");
            if (values is TensorProto tp)
                return tp;

            var origin_dtype = values.GetDataType();
            if (dtype == TF_DataType.DtInvalid)
                dtype = origin_dtype;
            else if(origin_dtype != dtype)
            {
                var new_system_dtype = dtype.as_system_dtype();
                if (values is long[] long_values)
                {
                    if (dtype == TF_DataType.TF_INT32)
                        values = long_values.Select(x => (int)Convert.ChangeType(x, new_system_dtype)).ToArray();
                }
                else
                    values = Convert.ChangeType(values, new_system_dtype);

                dtype = values.GetDataType();
            }

            shape = shape ?? values.GetShape();
            var tensor_proto = new TensorProto
            {
                Dtype = dtype.as_datatype_enum(),
                TensorShape = shape.as_shape_proto()
            };

            if (values is NDArray nd)
            {
                // scalar
                if (nd.shape.IsScalar)
                {
                    switch (nd.dtype)
                    {
                        case TF_DataType.TF_BOOL:
                            tensor_proto.BoolVal.AddRange(nd.ToArray<bool>());
                            break;
                        case TF_DataType.TF_UINT8:
                            tensor_proto.IntVal.AddRange(nd.ToArray<byte>().Select(x => (int)x).ToArray());
                            break;
                        case TF_DataType.TF_INT32:
                            tensor_proto.IntVal.AddRange(nd.ToArray<int>());
                            break;
                        case TF_DataType.TF_INT64:
                            tensor_proto.Int64Val.AddRange(nd.ToArray<long>());
                            break;
                        case TF_DataType.TF_FLOAT:
                            tensor_proto.FloatVal.AddRange(nd.ToArray<float>());
                            break;
                        case TF_DataType.TF_DOUBLE:
                            tensor_proto.DoubleVal.AddRange(nd.ToArray<double>());
                            break;
                        default:
                            throw new Exception("make_tensor_proto Not Implemented");
                    }
                }
                else
                {
                    var len = nd.dtypesize * nd.size;
                    byte[] bytes = nd.ToByteArray();
                    tensor_proto.TensorContent = Google.Protobuf.ByteString.CopyFrom(bytes);
                }
            }
            else if (dtype == TF_DataType.TF_STRING && !(values is NDArray))
            {
                if (values is string str)
                    tensor_proto.StringVal.Add(Google.Protobuf.ByteString.CopyFromUtf8(str));
                else if (values is string[] str_values)
                    tensor_proto.StringVal.AddRange(str_values.Select(x => Google.Protobuf.ByteString.CopyFromUtf8(x)));
                else if (values is byte[] byte_values)
                    tensor_proto.TensorContent = Google.Protobuf.ByteString.CopyFrom(byte_values);
            }
            else if (values is Array array)
            {
                // array
                var len = dtype.get_datatype_size() * (int)shape.size;
                byte[] bytes = new byte[len];
                System.Buffer.BlockCopy(array, 0, bytes, 0, len);
                tensor_proto.TensorContent = Google.Protobuf.ByteString.CopyFrom(bytes);
            }
            else
            {
                switch (values)
                {
                    case Axis val:
                        tensor_proto.IntVal.AddRange(val.axis);
                        break;
                    case bool val:
                        tensor_proto.BoolVal.AddRange(new[] { val });
                        break;
                    case sbyte val:
                        tensor_proto.IntVal.AddRange(new[] { (int)val });
                        break;
                    case int val:
                        tensor_proto.IntVal.AddRange(new[] { val });
                        break;
                    case long val:
                        tensor_proto.Int64Val.AddRange(new[] { val });
                        break;
                    case float val:
                        tensor_proto.FloatVal.AddRange(new[] { val });
                        break;
                    case double val:
                        tensor_proto.DoubleVal.AddRange(new[] { val });
                        break;
                    default:
                        throw new Exception("make_tensor_proto Not Implemented");
                }
            }

            return tensor_proto;
        }

        public static Shape constant_value_as_shape(Tensor tensor)
        {
            bool hasattr(Graph property, string attr)
            {
                var t = property.GetType().GetProperties();
                foreach (System.Reflection.PropertyInfo pi in t)
                {
                    if (pi.Name == attr)
                        return true;
                }
                return false;
            }

            if (tensor.GetType() == typeof(EagerTensor))
            {
                if(tensor.dtype == TF_DataType.TF_INT64)
                    return new Shape(tensor.ToArray<long>());
                else
                    return new Shape(tensor.ToArray<int>());
            }

            if (tensor.shape.ndim == 0)
            {
                var value_ = constant_value(tensor);
                if (value_ == null)
                    throw new ValueError(
                        @"Received a scalar with unknown value as shape; require a statically
known scalar with value '-1' to describe an unknown shape.");
                if ((int)value_ != -1)
                    throw new ValueError(
                        String.Format(@"Received a scalar value {0} as shape; require a statically known
scalar with value '-1' to describe an unknown shape.", value_));
                return tensor.shape.unknown_shape(-1);
            }

            var shape = tensor.shape.with_rank(1);
            if (shape == new Shape(new int[] { 1 }))
            {
                return new Shape(new int[] { });
            }
            else if (tensor.op.type == "Cast")
            {
                var pre_cast = constant_value_as_shape(tensor.op.inputs[0]);
                if (pre_cast.dims == null)
                    return pre_cast;
                var cast_dtype = dtypes.as_tf_dtype((Type)tensor.op.get_attr("DstT"));
                if (!Array.Exists(new[] { dtypes.int32, dtypes.int64 }, cast_dtype_ => cast_dtype_ == cast_dtype))
                    return tensor.shape.unknown_shape((int)shape.dims[0]);

                long[] x_ = { };
                foreach (var x in pre_cast.dims)
                    if (x != -1)
                        x_[x_.Length] = x;
                    else
                        x_[x_.Length] = -1;
                var dest_dtype_shape_array = np.array(x_).astype(cast_dtype);

                long[] y_ = { };
                foreach (int y in dest_dtype_shape_array.ToArray<int>())
                    if (y >= 0)
                        y_[y_.Length] = y;
                    else
                        y_[y_.Length] = -1;
                return new Shape(y_);
            }
            else if (tensor.op.type == "Shape")
            {
                return tensor.op.inputs[0].shape;
            }
            else if (tensor.op.type == "Pack")
            {
                var ret_ = new Shape(new int[] { });
                if ((int)tensor.op.get_attr("axis") != 0)
                    throw new ValueError(String.Format(
                        @"Since rank 1 inputs are expected, Pack's axis: {0} must be 0, otherwise it
would not be rank 1.", tensor.op.get_attr("axis")));
                foreach (Tensor pack_input in tensor.op.inputs)
                {
                    var pack_input_val = (int)constant_value(pack_input);
                    Dimension new_dim;
                    if (pack_input_val < 0)
                    {
                        new_dim = new Dimension(-1);
                    }
                    else if (pack_input_val == null)
                    {
                        new_dim = new Dimension(-1);
                    }
                    else
                    {
                        new_dim = new Dimension(pack_input_val);
                    }
                    ret_ = ret_.concatenate(new long[] { new_dim });
                }
                return ret_;
            }
            else if (tensor.op.type == "Concat")
            {
                var ret_ = new Shape(new int[] { });

                var inputlist_ = new ArraySegment<Tensor>(tensor.op.inputs, 1,
                                                        tensor.op.inputs.Length - 1);
                foreach (var concat_input in inputlist_)
                {
                    ret_ = ret_.concatenate(constant_value_as_shape(concat_input));
                }
                return ret_;
            }
            else if (tensor.op.type == "StridedSlice")
            {
                try
                {
                    var begin = constant_value(tensor.op.inputs[1]);
                    var end = constant_value(tensor.op.inputs[2]);
                    var strides = constant_value(tensor.op.inputs[3]);
                    if (new[] { begin, end, strides }.All(x => x == null))
                    {
                        begin = begin[0];
                        end = end[0];
                        strides = strides[0];
                        var begin_mask = tensor.op.get_attr("begin_mask");
                        if ((int)begin_mask == 1)
                        {
                            begin = null;
                        }
                        var end_mask = tensor.op.get_attr("end_mask");
                        if ((int)end_mask == 1)
                        {
                            end = null;
                        }

                        var ellipsis_mask = tensor.op.get_attr("ellipsis_mask");
                        var new_axis_mask = tensor.op.get_attr("new_axis_mask");
                        var shrink_axis_mask = tensor.op.get_attr("shrink_axis_mask");

                        bool valid_attributes;
                        if (!(bool)ellipsis_mask && !(bool)new_axis_mask &&
                            !(bool)shrink_axis_mask && !((bool)begin_mask || (int)begin_mask == 1) &&
                            !((bool)end_mask || (int)end_mask == 1))
                        {
                            valid_attributes = true;
                        }
                        else { valid_attributes = false; }
                        if (valid_attributes)
                        {
                            // sorry for the mess here, but this hacky solution was the best way
                            // i could come up with to implement the things done in python in c#
                            var prev_ = constant_value_as_shape(tensor.op.inputs[0]).dims;
                            var prev = prev_.Skip((int)begin).Take((int)end - (int)begin).ToArray();
                            // 100 being the comparison doesn't really matter here; it's going to break anyway
                            for (int iter = 0; iter != 100; iter = iter + (int)strides)
                            {
                                prev[prev.Length] = prev_[iter];
                                if ((iter + (int)strides) > prev_.Length)
                                    break;
                            }
                            var ret_ = new Shape(prev);
                            return ret_;
                        }
                    }
                }
                catch (Exception ex)
                {
                    if (ex is ValueError || ex is TypeError) { }
                }
            }
            else if (tensor.op.type == "Placeholder" &&
                  tensor.op.graph.building_function &&
                  tensor.op.graph is FuncGraph func_graph)
            {
                int i = 0;
                foreach (Tensor capture in func_graph.internal_captures)
                {
                    if (capture.GetType() == typeof(Tensor))
                    {
                        var external_capture = func_graph.external_captures[i];
                        return constant_value_as_shape(external_capture);
                    }

                    i++;
                }
            }

            var ret = tensor.shape.unknown_shape((int)shape.dims[0]);
            var value = constant_value(tensor);
            if (!(value is null))
            {
                var d_ = new int[value.size];
                foreach (var (index, d) in enumerate(value.ToArray<int>()))
                    d_[index] = d >= 0 ? d : -1;
                
                ret = ret.merge_with(new Shape(d_));
            }
            return ret;
        }

        public static TensorShapeProto as_shape<T>(T[] dims)
        {
            TensorShapeProto shape = new TensorShapeProto();

            for (int i = 0; i < dims.Length; i++)
            {
                var dim = new TensorShapeProto.Types.Dim();
                switch (dims[i])
                {
                    case int n:
                        dim.Size = n;
                        break;
                    case long l:
                        dim.Size = l;
                        break;
                    default:
                        throw new NotImplementedException("as_shape Not Implemented");
                }
                // dim.Name = $"dim_{i}";

                shape.Dim.Add(dim);
            }

            return shape;
        }

        public static Shape to_shape(long[] dims)
        {
            return new Shape(dims.Select(x => (int)x).ToArray());
        }

        public static Shape to_shape(int[] dims)
        {
            return new Shape(dims);
        }

        public static TensorShapeProto as_shape_proto(this Shape tshape)
        {
            TensorShapeProto shape = new TensorShapeProto();

            for (int i = 0; i < tshape.ndim; i++)
            {
                var dim = new TensorShapeProto.Types.Dim();
                dim.Size = tshape.dims[i];
                //dim.Name = $"dim_{i}";

                shape.Dim.Add(dim);
            }

            return shape;
        }

        public static Shape reshape(this Shape shape, int[] dims)
        {
            return new Shape(dims);
        }

        public static TensorShapeProto as_proto(this Shape tshape)
        {
            TensorShapeProto shape = new TensorShapeProto();

            for (int i = 0; i < tshape.ndim; i++)
            {
                var dim = new TensorShapeProto.Types.Dim();
                dim.Size = tshape.dims[i];
                //dim.Name = $"dim_{i}";

                shape.Dim.Add(dim);
            }

            return shape;
        }

        public static Tensor shape_tensor(int[] shape)
        {
            return ops.convert_to_tensor(shape, dtype: TF_DataType.TF_INT32, name: "shape");
        }

        public static ParsedSliceArgs ParseSlices(Slice[] slices)
        {
            var begin = new List<int>();
            var end = new List<int>();
            var strides = new List<int>();

            var index = 0;
            var (new_axis_mask, shrink_axis_mask) = (0, 0);
            var (begin_mask, end_mask) = (0, 0);
            var ellipsis_mask = 0;

            foreach (var s in slices)
            {
                if (s.IsNewAxis)
                {
                    begin.Add(0);
                    end.Add(0);
                    strides.Add(1);
                    new_axis_mask |= (1 << index);
                }
                else if (s.IsEllipsis)
                {
                    begin.Add(0);
                    end.Add(0);
                    strides.Add(1);
                    ellipsis_mask |= (1 << index);
                }
                else
                {
                    if (s.Start.HasValue)
                    {
                        begin.Add(s.Start.Value);
                    }
                    else
                    {
                        begin.Add(0);
                        begin_mask |= (1 << index);
                    }

                    if (s.Stop.HasValue)
                    {
                        end.Add(s.Stop.Value);
                    }
                    else
                    {
                        end.Add(0);
                        end_mask |= (1 << index);
                    }

                    strides.Add(s.Step);
                    if (s.IsIndex)
                        shrink_axis_mask |= (1 << index);
                }

                index += 1;
            }

            return new ParsedSliceArgs
            {
                Begin = begin.ToArray(),
                End = end.ToArray(),
                Strides = strides.ToArray(),
                BeginMask = begin_mask,
                EndMask = end_mask,
                EllipsisMask = ellipsis_mask,
                ShrinkAxisMask = shrink_axis_mask,
                NewAxisMask = new_axis_mask
            };
        }

        public static ParsedSliceArgs ParseSlices(Tensor start, Tensor stop = null, Tensor step = null)
        {
            var begin = new List<Tensor>();
            var end = new List<Tensor>();
            var strides = new List<Tensor>();

            var index = 0;
            var (new_axis_mask, shrink_axis_mask) = (0, 0);
            var (begin_mask, end_mask) = (0, 0);
            var ellipsis_mask = 0;

            begin.Add(start);

            if (stop == null)
                end.Add(start + 1);
            else
                end.Add(stop);

            shrink_axis_mask |= (1 << index);

            if (step == null)
                strides.Add(tf.constant(1, dtype: start.dtype));
            else
                strides.Add(step);

            return new ParsedSliceArgs
            {
                PackedBegin = array_ops.stack(begin),
                PackedEnd = array_ops.stack(end),
                PackedStrides = array_ops.stack(strides),
                BeginMask = begin_mask,
                EndMask = end_mask,
                EllipsisMask = ellipsis_mask,
                ShrinkAxisMask = shrink_axis_mask,
                NewAxisMask = new_axis_mask
            };
        }
    }
}
