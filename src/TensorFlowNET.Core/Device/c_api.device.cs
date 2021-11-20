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
using Tensorflow.Device;
using Tensorflow.Eager;
using Tensorflow.Util;

namespace Tensorflow
{
    public partial class c_api
    {
        /// <summary>
        /// Specify the device for `desc`.  Defaults to empty, meaning unconstrained.
        /// </summary>
        /// <param name="desc"></param>
        /// <param name="device"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_SetDevice(IntPtr desc, string device);

        /// <summary>
        /// Counts the number of elements in the device list.
        /// </summary>
        /// <param name="list">TF_DeviceList*</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern int TF_DeviceListCount(SafeDeviceListHandle list);

        /// <summary>
        /// Retrieves the type of the device at the given index.
        /// </summary>
        /// <param name="list">TF_DeviceList*</param>
        /// <param name="index">int</param>
        /// <param name="status">TF_Status*</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_DeviceListType(SafeDeviceListHandle list, int index, SafeStatusHandle status);

        /// <summary>
        /// Deallocates the device list.
        /// </summary>
        /// <param name="list">TF_DeviceList*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_DeleteDeviceList(IntPtr list);

        /// <summary>
        /// Create a new TFE_TensorHandle with the same contents as 'h' but placed
        /// in the memory of the device name 'device_name'.
        /// </summary>
        /// <param name="h">TFE_TensorHandle*</param>
        /// <param name="ctx">TFE_Context*</param>
        /// <param name="device_name">char*</param>
        /// <param name="status">TF_Status*</param>
        /// <returns>TFE_TensorHandle*</returns>
        [DllImport(TensorFlowLibName)]
        public static extern SafeEagerTensorHandle TFE_TensorHandleCopyToDevice(SafeEagerTensorHandle h, SafeContextHandle ctx, string device_name, SafeStatusHandle status);

        /// <summary>
        /// Retrieves the full name of the device (e.g. /job:worker/replica:0/...)
        /// </summary>
        /// <param name="list">TF_DeviceList*</param>
        /// <param name="index"></param>
        /// <param name="status">TF_Status*</param>
        public static string TF_DeviceListName(SafeDeviceListHandle list, int index, SafeStatusHandle status)
        {
            using var _ = list.Lease();
            return StringPiece(TF_DeviceListNameImpl(list, index, status));
        }

        /// <summary>
        /// Retrieves the full name of the device (e.g. /job:worker/replica:0/...)
        /// The return value will be a pointer to a null terminated string. The caller
        /// must not modify or delete the string. It will be deallocated upon a call to
        /// TF_DeleteDeviceList.
        /// </summary>
        /// <param name="list">TF_DeviceList*</param>
        /// <param name="index"></param>
        /// <param name="status">TF_Status*</param>
        [DllImport(TensorFlowLibName, EntryPoint = "TF_DeviceListName")]
        private static extern IntPtr TF_DeviceListNameImpl(SafeDeviceListHandle list, int index, SafeStatusHandle status);
    }
}
