using OneOf.Types;
using System;
using Tensorflow.Gradients;
using Tensorflow.Util;
using static Tensorflow.Binding;

namespace Tensorflow.Eager
{
    /// <summary>
    /// python\eager\pywrap_tfe_src.cc
    /// </summary>
    public partial class EagerRunner
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="tape"></param>
        /// <param name="target"></param>
        /// <param name="sources"></param>
        /// <param name="output_gradients"></param>
        /// <param name="unconnected_gradients">determines the value returned if the target and
        /// sources are unconnected.When 'none' the value returned is None wheras when
        /// 'zero' a zero tensor in the same shape as the sources is returned.</param>
        /// <returns></returns>
        /// <exception cref="RuntimeError"></exception>
        public Tensor[] TFE_TapeGradient(ITape tape,
            Tensor[] target,
            Tensor[] sources,
            List<Tensor> output_gradients, 
            Tensor[] sources_raw,
            string unconnected_gradients)
        {
            if (!tape.Persistent)
            {
                var tape_set = tf.GetTapeSet();
                if (tape_set.Contains(tape))
                {
                    throw new RuntimeError("gradient() cannot be invoked within the " +
                        "GradientTape context (i.e., while operations are being " +
                        "recorded). Either move the call to gradient() to be " +
                        "outside the 'with tf.GradientTape' block, or " +
                        "use a persistent tape: " +
                        "'with tf.GradientTape(persistent=true)'");
                }
            }

            var target_vec = MakeTensorIDList(target);
            var sources_vec = MakeTensorIDList(sources);
            HashSet<long> sources_set = new HashSet<long>(sources_vec);
            var source_tensors_that_are_targets = new UnorderedMap<long, TapeTensor>();

            int len = target.Length;
            for(int i = 0; i < len; i++)
            {
                var target_id = target_vec[i];
                if (sources_set.Contains(target_id))
                {
                    var tensor = target[i];
                    source_tensors_that_are_targets[target_id] = TapeTensorFromTensor(tensor);
                }
            }

            List<Tensor> outgrad_vec = new();
            if(output_gradients is not null)
            {
                outgrad_vec = output_gradients.ToList();
            }
            var result = tape.ComputeGradient(target_vec, sources_vec, source_tensors_that_are_targets, outgrad_vec, true);


            bool unconnected_gradients_zero = unconnected_gradients == "zero";
            Tensor[] sources_obj = null;
            if (unconnected_gradients_zero)
            {
                sources_obj = MakeTensorList(sources_raw);
            }

            if (result.Length > 0)
            {
                for(int i = 0; i < result.Length; i++)
                {
                    if (result[i] is null && unconnected_gradients_zero)
                    {
                        var dtype = sources_obj[i].dtype;
                        result[i] = new TapeTensor(sources_vec[i], dtype, sources_obj[i]).ZerosLike();
                    }
                }
            }
            return result;
        }

        Tensor[] MakeTensorList(IEnumerable<Tensor> tensors)
        {
            return tensors.ToArray();
        }

        long[] MakeTensorIDList(Tensor[] tensors)
        {
            int len = tensors.Length;
            long[] ids = new long[len];
            for(int i = 0; i < len; i++)
            {
                var tensor = tensors[i];
                ids[i] = tensor.Id;
            }
            return ids;
        }

        TF_DataType[] MakeTensorDtypeList(Tensor[] tensors)
        {
            int len = tensors.Length;
            TF_DataType[] dtypes = new TF_DataType[len];
            for (int i = 0; i < len; i++)
            {
                var tensor = tensors[i];
                dtypes[i] = tensor.dtype;
            }
            return dtypes;
        }

        TapeTensor TapeTensorFromTensor(Tensor tensor)
        {
            long id = tensor.Id;
            var dtype = tensor.dtype;
            if (tensor is EagerTensor)
            {
                var handle = tensor.EagerTensorHandle;
                if (DTypeNeedsHandleData(dtype))
                {
                    return new TapeTensor(id, c_api.TFE_TensorHandleDataType(handle), tensor);
                }

                Status status = new();
                int num_dims = c_api.TFE_TensorHandleNumDims(handle, status);
                long[] dims = new long[num_dims];
                for(int i = 0; i < num_dims; i++)
                {
                    dims[i] = c_api.TFE_TensorHandleDim(handle, i, status);
                }

                if(status.Code != TF_Code.TF_OK)
                {
                    return new TapeTensor(id, TF_DataType.DtInvalid, Shape.Null);
                }
                else
                {
                    Shape tensor_shape = new(dims);
                    return new TapeTensor(id, dtype, tensor_shape);
                }
            }
            var shape_tuple = tensor.shape.dims;
            if(ListContainNone(shape_tuple) || DTypeNeedsHandleData(dtype))
            {
                return new TapeTensor(id, dtype, tensor);
            }
            long[] l = new long[shape_tuple.Length];
            for(int i = 0; i < shape_tuple.Length; i++)
            {
                if (shape_tuple[i] < 0)
                {
                    l[i] = 0;
                }
                else
                {
                    l[i] = shape_tuple[i];
                }
            }
            return new TapeTensor(id, dtype, new Shape(l));
        }

        bool DTypeNeedsHandleData(TF_DataType dtype)
        {
            return dtype == dtypes.variant || dtype == dtypes.resource;
        }

        bool ListContainNone(long[]? list)
        {
            if(list is null)
            {
                return true;
            }
            int len = list.Length;
            if(len == 0)
            {
                return true;
            }
            for(int i = 0; i < len; i++)
            {
                if (list[i] == -1)
                {
                    return true;
                }
            }
            return false;
        }
    }
}
