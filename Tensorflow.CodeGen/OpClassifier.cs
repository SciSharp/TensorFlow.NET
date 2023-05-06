using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.RegularExpressions;

namespace Tensorflow.CodeGen
{
    public class OpClassifier
    {
        private static readonly string _filenamePattern = @"^gen_[a-z]*_ops.py$";
        private static readonly string _pythonFunctionPattern = @"def\s+(\w+)\((?:\s*\w+\s*(?:=\s*[\S]*)*,\s*)*\s*\w+\s*=None\s*\):";
        private Dictionary<string, HashSet<string>> _opSet = new();
        public Dictionary<string, HashSet<string>> OpSet => _opSet;
        public OpClassifier(string pythonFileFolder)
        {
            DirectoryInfo directory = new DirectoryInfo(pythonFileFolder);

            foreach (FileInfo file in directory.GetFiles())
            {
                if (Regex.IsMatch(file.Name, _filenamePattern))
                {
                    string filenamePrefix = file.Name.Split('.')[0];
                    string content = File.ReadAllText(file.FullName);
                    var matches = Regex.Matches(content, _pythonFunctionPattern);
                    foreach(Match match in matches)
                    {
                        var funcName = match.Groups[1].Value;
                        if (!funcName.EndsWith("_eager_fallback"))
                        {
                            _opSet.SetDefault(filenamePrefix, new HashSet<string>()).Add(funcName);
                        }
                    }
                }
            }
        }
    }
}
