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

using static Tensorflow.Binding;

namespace Tensorflow
{
    public class check_ops
    {
        /// <summary>
        /// Assert the condition `x == y` holds element-wise.
        /// </summary>
        /// <param name="t1"></param>
        /// <param name="t2"></param>
        /// <param name="name"></param>
        public static Operation assert_equal<T1, T2>(T1 t1, T2 t2, object[] data = null, string message = null, string name = null)
        {
            if (message == null)
                message = "";

            return tf_with(ops.name_scope(name, "assert_equal", new { t1, t2, data }), delegate
            {
                var x = ops.convert_to_tensor(t1, name: "x");
                var y = ops.convert_to_tensor(t2, name: "y");

                if (data == null)
                {
                    data = new object[]
                    {
                        message,
                        "Condition x == y did not hold element-wise:",
                        $"x (%s) = {x.name}",
                        x,
                        $"y (%s) = {y.name}",
                        y
                    };
                }

                var eq = gen_math_ops.equal(x, y);
                var condition = math_ops.reduce_all(eq);
                var x_static = tensor_util.constant_value(x);
                var y_static = tensor_util.constant_value(y);
                return control_flow_ops.Assert(condition, data);
            });
        }

        public static Operation assert_greater_equal(Tensor x, Tensor y, object[] data = null, string message = null,
            string name = null)
        {
            if (message == null)
                message = "";

            return tf_with(ops.name_scope(name, "assert_greater_equal", new { x, y, data }), delegate
              {
                  x = ops.convert_to_tensor(x, name: "x");
                  y = ops.convert_to_tensor(y, name: "y");
                  string x_name = x.name;
                  string y_name = y.name;
                  if (data == null)
                  {
                      data = new object[]
                      {
                        message,
                        "Condition x >= y did not hold element-wise:",
                        $"x (%s) = {x_name}",
                        x,
                        $"y (%s) = {y_name}",
                        y
                      };
                  }

                  var condition = math_ops.reduce_all(gen_math_ops.greater_equal(x, y));
                  return control_flow_ops.Assert(condition, data);
              });
        }


        public static Operation assert_positive(Tensor x, object[] data = null, string message = null, string name = null)
        {
            if (message == null)
                message = "";

            return tf_with(ops.name_scope(name, "assert_positive", new { x, data }), delegate
            {
                x = ops.convert_to_tensor(x, name: "x");
                if (data == null)
                {
                    name = x.name;
                    data = new object[]
                    {
                        message,
                        "Condition x > 0 did not hold element-wise:",
                        $"x (%s) = {name}",
                        x
                    };
                }
                var zero = ops.convert_to_tensor(0, dtype: x.dtype);
                return assert_less(zero, x, data: data);
            });
        }

        public static Operation assert_less(Tensor x, Tensor y, object[] data = null, string message = null, string name = null)
        {
            if (message == null)
                message = "";

            return tf_with(ops.name_scope(name, "assert_less", new { x, y, data }), delegate
            {
                x = ops.convert_to_tensor(x, name: "x");
                y = ops.convert_to_tensor(y, name: "y");
                string x_name = x.name;
                string y_name = y.name;
                if (data == null)
                {
                    data = new object[]
                    {
                        message,
                        "Condition x < y did not hold element-wise:",
                        $"x (%s) = {x_name}",
                        $"y (%s) = {y_name}",
                        y
                    };
                }
                var condition = math_ops.reduce_all(gen_math_ops.less(x, y));
                return control_flow_ops.Assert(condition, data);
            });
        }
    }
}
