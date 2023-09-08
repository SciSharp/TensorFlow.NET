/*****************************************************************************
   Copyright 2021 Haiping Chen. All Rights Reserved.

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

using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Keras.Saving.Common;

namespace Tensorflow
{
    [JsonConverter(typeof(CustomizedAxisJsonConverter))]
    public class Axis
    {
        public int[] axis { get; set; }
        public int size => axis == null ? -1 : axis.Length;
        public bool IsScalar { get; init; }

        public int this[int index] => axis[index];

        public Axis(params int[] axis)
        {
            this.axis = axis;
        }

        public static implicit operator int[]?(Axis axis)
            => axis?.axis;

        public static implicit operator int(Axis axis)
            => axis.axis[0];

        public static implicit operator Axis(int axis)
            => new Axis(axis) { IsScalar = true };

        public static implicit operator Axis((int, int) axis)
            => new Axis(axis.Item1, axis.Item2);

        public static implicit operator Axis((int, int, int) axis)
            => new Axis(axis.Item1, axis.Item2, axis.Item3);

        public static implicit operator Axis(int[] axis)
            => new Axis(axis);

        public static implicit operator Axis(long[] axis)
            => new Axis(axis.Select(x => (int)x).ToArray());

        public static implicit operator Axis(Shape axis)
            => new Axis(axis.dims.Select(x => (int)x).ToArray());

        public static implicit operator Tensor(Axis axis)
            => constant_op.constant(axis);

        public static bool operator ==(Axis left, int right)
            => left.IsScalar && left[0] == right;

        public static bool operator !=(Axis left, int right)
            => !(left == right);

        public override string ToString()
            => IsScalar ? $"{axis[0]}" : $"({string.Join(", ", axis)})";
    }
}
