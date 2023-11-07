using Google.Protobuf;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Eager;
using static Tensorflow.CppShapeInferenceResult.Types;

namespace Tensorflow.Operations
{
    public static class handle_data_util
    {
        public static void copy_handle_data(Tensor source_t, Tensor target_t)
        {
            if(target_t.dtype == dtypes.resource || target_t.dtype == dtypes.variant)
            {
                HandleData handle_data;
                if(source_t is EagerTensor)
                {
                    handle_data = source_t.HandleData;
                }
                else
                {
                    handle_data = ops.get_resource_handle_data(source_t);
                }
                if(handle_data is not null && handle_data.IsSet && handle_data.ShapeAndType is not null 
                    && handle_data.ShapeAndType.Count > 0)
                {
                    set_handle_data(target_t, handle_data);
                }
            }
        }

        public static HandleData create_handle_data(Shape shape, TF_DataType dtype)
        {
            HandleData handle_data = new();
            handle_data.IsSet = true;
            handle_data.ShapeAndType.Add(new HandleShapeAndType()
            {
                Shape = shape.as_proto(),
                Dtype = dtype.as_datatype_enum()
            });
            return handle_data;
        }

        public static void set_handle_data(Tensor target_t, HandleData handle_data)
        {
            if(target_t is EagerTensor)
            {
                target_t.HandleData = handle_data;
                return;
            }
            Status status = new();
            var proto = handle_data.ToByteArray();
            c_api.TF_SetHandleShapeAndType(target_t.graph.c_graph, target_t._as_tf_output(), proto, proto.Length, status);
            status.Check(true);
        }

        public static HandleData get_resource_handle_data(Tensor graph_op) => ops.get_resource_handle_data(graph_op);
    }
}
