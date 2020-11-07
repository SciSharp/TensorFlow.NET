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

using NumSharp;

namespace Tensorflow
{
    public partial class ResourceVariable
    {
        public static Tensor operator +(ResourceVariable x, int y) => x.value() + y;
        public static Tensor operator +(ResourceVariable x, float y) => x.value() + y;
        public static Tensor operator +(ResourceVariable x, double y) => x.value() + y;
        public static Tensor operator +(ResourceVariable x, ResourceVariable y) => x.value() + y.value();
        public static Tensor operator -(ResourceVariable x, int y) => x.value() - y;
        public static Tensor operator -(ResourceVariable x, float y) => x.value() - y;
        public static Tensor operator -(ResourceVariable x, double y) => x.value() - y;
        public static Tensor operator -(ResourceVariable x, Tensor y) => x.value() - y;
        public static Tensor operator -(ResourceVariable x, ResourceVariable y) => x.value() - y.value();

        public static Tensor operator *(ResourceVariable x, ResourceVariable y) => x.value() * y.value();
        public static Tensor operator *(ResourceVariable x, NDArray y) => x.value() * y;

        public static Tensor operator <(ResourceVariable x, Tensor y) => x.value() < y;

        public static Tensor operator >(ResourceVariable x, Tensor y) => x.value() > y;
    }
}
