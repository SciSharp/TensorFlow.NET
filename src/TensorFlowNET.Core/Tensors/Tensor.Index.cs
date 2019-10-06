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
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public partial class Tensor
    {
        public Tensor this[int idx] => slice(idx);

        public Tensor this[params Slice[] slices]
        {
            get
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
                    }

                    index += 1;
                }

                return tf_with(ops.name_scope(null, "strided_slice", new { begin, end, strides }), scope =>
                {
                    string name = scope;
                    if (begin != null)
                    {
                        var (packed_begin, packed_end, packed_strides) =
                            (array_ops.stack(begin.ToArray()),
                                array_ops.stack(end.ToArray()),
                                array_ops.stack(strides.ToArray()));

                        return gen_array_ops.strided_slice(
                            this,
                            packed_begin,
                            packed_end,
                            packed_strides,
                            begin_mask: begin_mask,
                            end_mask: end_mask,
                            shrink_axis_mask: shrink_axis_mask,
                            new_axis_mask: new_axis_mask,
                            ellipsis_mask: ellipsis_mask,
                            name: name);
                    }

                    throw new NotImplementedException("");
                });
            }
        }

        public Tensor this[params string[] slices] 
            => this[slices.Select(x => new Slice(x)).ToArray()];


        public Tensor slice(Slice slice)
        {
            var slice_spec = new int[] { slice.Start.Value };
            var begin = new List<int>();
            var end = new List<int>();
            var strides = new List<int>();

            var index = 0;
            var (new_axis_mask, shrink_axis_mask) = (0, 0);
            var (begin_mask, end_mask) = (0, 0);
            var ellipsis_mask = 0;

            foreach (var s in slice_spec)
            {
                begin.Add(s);
                if (slice.Stop.HasValue)
                {
                    end.Add(slice.Stop.Value);
                }
                else
                {
                    end.Add(0);
                    end_mask |= (1 << index);
                }

                strides.Add(slice.Step);

                index += 1;
            }

            return tf_with(ops.name_scope(null, "strided_slice", new { begin, end, strides }), scope =>
            {
                string name = scope;
                if (begin != null)
                {
                    var (packed_begin, packed_end, packed_strides) =
                        (array_ops.stack(begin.ToArray()),
                            array_ops.stack(end.ToArray()),
                            array_ops.stack(strides.ToArray()));

                    return gen_array_ops.strided_slice(
                        this,
                        packed_begin,
                        packed_end,
                        packed_strides,
                        begin_mask: begin_mask,
                        end_mask: end_mask,
                        shrink_axis_mask: shrink_axis_mask,
                        new_axis_mask: new_axis_mask,
                        ellipsis_mask: ellipsis_mask,
                        name: name);
                }

                throw new NotImplementedException("");
            });
        }

        public Tensor slice(int start)
        {
            var slice_spec = new int[] { start };
            var begin = new List<int>();
            var end = new List<int>();
            var strides = new List<int>();

            var index = 0;
            var (new_axis_mask, shrink_axis_mask) = (0, 0);
            var (begin_mask, end_mask) = (0, 0);
            var ellipsis_mask = 0;

            foreach (var s in slice_spec)
            {
                begin.Add(s);
                end.Add(s + 1);
                strides.Add(1);
                shrink_axis_mask |= (1 << index);
                index += 1;
            }

            return tf_with(ops.name_scope(null, "strided_slice", new { begin, end, strides }), scope =>
            {
                string name = scope;
                if (begin != null)
                {
                    var (packed_begin, packed_end, packed_strides) =
                        (array_ops.stack(begin.ToArray()),
                            array_ops.stack(end.ToArray()),
                            array_ops.stack(strides.ToArray()));

                    return gen_array_ops.strided_slice(
                        this,
                        packed_begin,
                        packed_end,
                        packed_strides,
                        begin_mask: begin_mask,
                        end_mask: end_mask,
                        shrink_axis_mask: shrink_axis_mask,
                        new_axis_mask: new_axis_mask,
                        ellipsis_mask: ellipsis_mask,
                        name: name);
                }

                throw new NotImplementedException("");
            });
        }
    }
}
