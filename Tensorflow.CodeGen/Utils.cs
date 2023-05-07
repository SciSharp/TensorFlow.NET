using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata.Ecma335;
using System.Text;
using System.Threading.Tasks;

namespace Tensorflow.CodeGen
{
    public static class Utils
    {
        public static string ConvertToUnderscore(string input)
        {
            if (string.IsNullOrEmpty(input))
            {
                return input;
            }

            StringBuilder result = new StringBuilder();

            int state = 1; // the previous char was not lowered.
            for (int i = 0; i < input.Length; i++)
            {
                char current = input[i];

                // 首字母不需要添加下划线
                if (char.IsUpper(current))
                {
                    if(i > 0)
                    {
                        char pre = input[i - 1];
                        if (char.IsDigit(pre))
                        {
                            result.Append(char.ToLower(current));
                            continue;
                        }
                    }
                    if (state == 0)
                    {
                        result.Append("_");
                        state = 1;
                    }
                    result.Append(char.ToLower(current));
                }
                else
                {
                    result.Append(char.ToLower(current));
                    state = 0;
                }
            }

            return result.ToString();
        }
    }
}
