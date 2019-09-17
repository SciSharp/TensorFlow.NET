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

namespace Tensorflow
{
    public partial class tensorflow
    {
        /// <summary>
        /// Assert the condition `x == y` holds element-wise.
        /// </summary>
        /// <typeparam name="T1"></typeparam>
        /// <typeparam name="T2"></typeparam>
        /// <param name="t1"></param>
        /// <param name="t2"></param>
        /// <param name="data"></param>
        /// <param name="message"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor assert_equal<T1, T2>(T1 t1,
            T2 t2,
            object[] data = null,
            string message = null,
            string name = null)
            => check_ops.assert_equal(t1, 
                t2, 
                data: data, 
                message: message, 
                name: name);

    }
}
