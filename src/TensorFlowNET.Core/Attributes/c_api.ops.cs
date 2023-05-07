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
using System.Runtime.InteropServices;

namespace Tensorflow
{
    public partial class c_api
    {
        /// <summary>
        /// Fills in `value` with the value of the attribute `attr_name`.  `value` must
        /// point to an array of length at least `max_length` (ideally set to
        /// TF_AttrMetadata.total_size from TF_OperationGetAttrMetadata(oper,
        /// attr_name)).
        /// </summary>
        /// <param name="oper">TF_Operation*</param>
        /// <param name="attr_name">const char*</param>
        /// <param name="status">TF_Status*</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern TF_AttrMetadata TF_OperationGetAttrMetadata(IntPtr oper, string attr_name, SafeStatusHandle status);

        /// <summary>
        /// Fills in `value` with the value of the attribute `attr_name`.  `value` must
        /// point to an array of length at least `max_length` (ideally set to
        /// TF_AttrMetadata.total_size from TF_OperationGetAttrMetadata(oper,
        /// attr_name)). 
        /// </summary>
        /// <param name="oper">TF_Operation*</param>
        /// <param name="attr_name">const char*</param>
        /// <param name="value">void* </param>
        /// <param name="max_length">size_t</param>
        /// <param name="status">TF_Status*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_OperationGetAttrString(IntPtr oper, string attr_name, IntPtr value, uint max_length, SafeStatusHandle status);

        /// <summary>
        /// Sets `output_attr_value` to the binary-serialized AttrValue proto
        /// representation of the value of the `attr_name` attr of `oper`.
        /// </summary>
        /// <param name="oper"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern int TF_OperationGetAttrValueProto(IntPtr oper, string attr_name, SafeBufferHandle output_attr_value, SafeStatusHandle status);

        [DllImport(TensorFlowLibName)]
        public static extern void TF_OperationGetAttrType(IntPtr oper, string attr_name, IntPtr value, SafeStatusHandle status);

        [DllImport(TensorFlowLibName)]
        public static extern void TF_OperationGetAttrInt(IntPtr oper, string attr_name, IntPtr value, SafeStatusHandle status);

        [DllImport(TensorFlowLibName)]
        public static extern void TF_OperationGetAttrFloat(IntPtr oper, string attr_name, IntPtr value, SafeStatusHandle status);

        [DllImport(TensorFlowLibName)]
        public static extern void TF_OperationGetAttrBool(IntPtr oper, string attr_name, IntPtr value, SafeStatusHandle status);

        [DllImport(TensorFlowLibName)]
        public static extern void TF_OperationGetAttrShape(IntPtr oper, string attr_name, long[] value, int num_dims, SafeStatusHandle status);

        [DllImport(TensorFlowLibName)]
        public static extern void TF_SetAttrBool(IntPtr desc, string attr_name, bool value);

        [DllImport(TensorFlowLibName)]
        public static extern void TF_SetAttrValueProto(IntPtr desc, string attr_name, byte[] proto, ulong proto_len, SafeStatusHandle status);

        /// <summary>
        /// Set `num_dims` to -1 to represent "unknown rank".
        /// </summary>
        /// <param name="desc"></param>
        /// <param name="attr_name"></param>
        /// <param name="dims"></param>
        /// <param name="num_dims"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_SetAttrShape(IntPtr desc, string attr_name, long[] dims, int num_dims);

        /// <summary>
        /// Call some TF_SetAttr*() function for every attr that is not
        /// inferred from an input and doesn't have a default value you wish to
        /// keep.
        /// 
        /// `value` must point to a string of length `length` bytes.
        /// </summary>
        /// <param name="desc">TF_OperationDescription*</param>
        /// <param name="attr_name">const char*</param>
        /// <param name="value">const void*</param>
        /// <param name="length">size_t</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_SetAttrString(IntPtr desc, string attr_name, string value, uint length);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="desc"></param>
        /// <param name="attr_name"></param>
        /// <param name="values"></param>
        /// <param name="lengths"></param>
        /// <param name="num_values"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_SetAttrStringList(IntPtr desc, string attr_name, IntPtr[] values, uint[] lengths, int num_values);

        [DllImport(TensorFlowLibName)]
        public static extern void TF_SetAttrTensor(IntPtr desc, string attr_name, SafeTensorHandle value, SafeStatusHandle status);

        [DllImport(TensorFlowLibName)]
        public static extern void TF_SetAttrType(IntPtr desc, string attr_name, TF_DataType value);
    }
}
