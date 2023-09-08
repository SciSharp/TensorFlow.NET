using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Eager;

namespace Tensorflow.Operations
{
    internal class list_ops
    {
        private static void _set_handle_data(Tensor list_handle, Shape element_shape, TF_DataType element_dtype)
        {
            if(list_handle is EagerTensor eagerTensor)
            {
                var handle_data = new CppShapeInferenceResult.Types.HandleData();
                handle_data.IsSet = true;
                handle_data.ShapeAndType.Add(new CppShapeInferenceResult.Types.HandleShapeAndType()
                {
                    Shape = element_shape.as_proto(),
                    Dtype = element_dtype.as_datatype_enum(),
                    Type = new FullTypeDef() { TypeId = FullTypeId.TftArray }
                });
                list_handle.HandleData = handle_data;
            }
        }

        private static Tensor _build_element_shape(Shape? shape)
        {
            if(shape is null || shape.IsNull)
            {
                return ops.convert_to_tensor(-1);
            }
            else
            {
                return ops.convert_to_tensor(shape, dtype: dtypes.int32);
            }
        }

        public static Tensor tensor_list_reserve(Shape? shape, Tensor num_elements, TF_DataType element_dtype, string name = null)
        {
            var result = gen_list_ops.tensor_list_reserve(_build_element_shape(shape), num_elements, element_dtype, name);
            _set_handle_data(result, shape, element_dtype);
            return result;
        }

        public static Tensor tensor_list_from_tensor(Tensor tensor, Shape element_shape, string? name = null)
        {
            var result = gen_list_ops.tensor_list_from_tensor(tensor, _build_element_shape(element_shape), name);
            _set_handle_data(result, tensor.shape, tensor.dtype);
            return result;
        }

        public static Tensor tensor_list_get_item(Tensor input_handle, Tensor index, TF_DataType element_dtype, 
            Shape? element_shape = null, string? name = null)
        {
            return gen_list_ops.tensor_list_get_item(input_handle, index, _build_element_shape(element_shape),
                element_dtype, name);
        }

        public static Tensor tensor_list_set_item(Tensor input_handle, Tensor index, Tensor item,
            bool resize_if_index_out_of_bounds = false, string? name = null)
        {
            if (resize_if_index_out_of_bounds)
            {
                var input_list_size = gen_list_ops.tensor_list_length(input_handle);
                input_handle = control_flow_ops.cond(index >= input_list_size,
                    () => gen_list_ops.tensor_list_resize(input_handle, index + 1),
                    () => input_handle);
            }
            var output_handle = gen_list_ops.tensor_list_set_item(input_handle, index, item, name);
            handle_data_util.copy_handle_data(input_handle, output_handle);
            return output_handle;
        }

        public static Tensor tensor_list_stack(Tensor input_handle, TF_DataType element_dtype, int num_elements = -1, 
            Shape? element_shape = null, string? name = null)
        {
            return gen_list_ops.tensor_list_stack(input_handle, _build_element_shape(element_shape), element_dtype, num_elements, name);
        }

        public static Tensor tensor_list_gather(Tensor input_handle, Tensor indices, TF_DataType element_dtype,
            Shape? element_shape = null, string? name = null)
        {
            return gen_list_ops.tensor_list_gather(input_handle, indices, _build_element_shape(element_shape), element_dtype, name);
        }

        public static Tensor tensor_list_scatter(Tensor tensor, Tensor indices, Shape? element_shape = null, Tensor? input_handle = null, 
            string? name = null)
        {
            if(input_handle is not null)
            {
                var output_handle = gen_list_ops.tensor_list_scatter_into_existing_list(input_handle, tensor, indices, name);
                handle_data_util.copy_handle_data(input_handle, output_handle);
                return output_handle;
            }
            else
            {
                var output_handle = gen_list_ops.tensor_list_scatter_v2(tensor, indices, _build_element_shape(element_shape), 
                    constant_op.constant(-1), name);
                _set_handle_data(output_handle, element_shape, tensor.dtype);
                return output_handle;
            }
        }

        public static Tensor empty_tensor_list(Shape? element_shape, TF_DataType element_dtype, int max_num_elements = -1,
            string? name = null)
        {
            return gen_list_ops.empty_tensor_list(_build_element_shape(element_shape), element_dtype: element_dtype,
                max_num_elements: ops.convert_to_tensor(max_num_elements, dtype: dtypes.int32), name: name);
        }
    }
}
