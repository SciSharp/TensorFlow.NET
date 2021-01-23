using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Linq;
using static Tensorflow.Binding;
using System.Text.RegularExpressions;

namespace Tensorflow
{
    public class Diagnostician
    {
        public void Diagnose(string log)
        {
            var lines = File.ReadAllLines(log);

            foreach(var (i, line) in enumerate(lines))
            {
                if(line.StartsWith("New Tensor "))
                {
                    var pointers = Regex.Matches(line, "0x[0-9a-f]{16}");
                    var tensorHandle = pointers[0].Value;
                    var tensorDataHandle = pointers[1].Value;

                    if (lines.Skip(i).Count(x => x.StartsWith("Delete Tensor ") 
                        && x.Contains(tensorHandle)
                        && x.Contains(tensorDataHandle)) == 0)
                        Console.WriteLine(line);
                }
                else if (line.StartsWith("New EagerTensorHandle "))
                {
                    var pointers = Regex.Matches(line, "0x[0-9a-f]{16}");
                    var tensorHandle = pointers[0].Value;

                    var del = $"Delete EagerTensorHandle {tensorHandle}";

                    if (lines.Skip(i).Count(x => x == del) == 0)
                        Console.WriteLine(line);
                }
                else if (line.StartsWith("Take EagerTensorHandle "))
                {
                    var pointers = Regex.Matches(line, "0x[0-9a-f]{16}");
                    var eagerTensorHandle = pointers[0].Value;
                    var tensorHandle = pointers[1].Value;

                    var delTensor = $"Delete Tensor {tensorHandle}";
                    var delEagerTensor = $"Delete EagerTensorHandle {eagerTensorHandle}";
                    if (lines.Skip(i).Count(x => x.StartsWith(delTensor)) == 0
                        || lines.Skip(i).Count(x => x.StartsWith(delEagerTensor)) == 0)
                        Console.WriteLine(line);
                }
                else if (line.StartsWith("Created Resource "))
                {
                    var pointers = Regex.Matches(line, "0x[0-9a-f]{16}");
                    var eagerTensorHandle = pointers[0].Value;

                    var delTensor = $"Deleted Resource {eagerTensorHandle}";
                    if (lines.Skip(i).Count(x => x.StartsWith(delTensor)) == 0)
                        Console.WriteLine(line);
                }
            }
        }
    }
}
