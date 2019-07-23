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

using Newtonsoft.Json;
using System.Collections.Generic;

namespace TensorFlowNET.Examples.Utility
{
    public class PbtxtItem
    {
        public string name { get; set; }
        public int id { get; set; }
        public string display_name { get; set; }
    }
    public class PbtxtItems
    {
        public List<PbtxtItem> items { get; set; }
    }

    public class PbtxtParser
    {
        public static PbtxtItems ParsePbtxtFile(string filePath)
        {
            string line;
            string newText = "{\"items\":[";

            using (System.IO.StreamReader reader = new System.IO.StreamReader(filePath))
            {

                while ((line = reader.ReadLine()) != null)
                {
                    string newline = string.Empty;

                    if (line.Contains("{"))
                    {
                        newline = line.Replace("item", "").Trim();
                        //newText += line.Insert(line.IndexOf("=") + 1, "\"") + "\",";
                        newText += newline;
                    }
                    else if (line.Contains("}"))
                    {
                        newText = newText.Remove(newText.Length - 1);
                        newText += line;
                        newText += ",";
                    }
                    else
                    {
                        newline = line.Replace(":", "\":").Trim();
                        newline = "\"" + newline;// newline.Insert(0, "\"");
                        newline += ",";

                        newText += newline;
                    }

                }

                newText = newText.Remove(newText.Length - 1);
                newText += "]}";

                reader.Close();
            }

            PbtxtItems items = JsonConvert.DeserializeObject<PbtxtItems>(newText);

            return items;
        }
    }
}
