using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace Tensorflow.Device
{
    public class DeviceSpec
    {
        private static ConcurrentDictionary<string, Components> _STRING_TO_COMPONENTS_CACHE = new();
        private static ConcurrentDictionary<Components, string> _COMPONENTS_TO_STRING_CACHE = new();
        private string _job;
        private int _replica;
        private int _task;
        private string _device_type;
        private int _device_index;
        private string _as_string;

        public string Job => _job;
        public int Replica => _replica;
        public int Task => _task;
        public string DeviceType => _device_type;
        public int DeviceIndex => _device_index;

        public DeviceSpec(string job = null, int replica = -1, int task = -1,
            string device_type = null, int device_index = -1)
        {
            _job = job;
            _replica = replica;
            _task = task;
            _device_type = device_type;
            _device_index = device_index;
            _as_string = _components_to_string(job, replica, task, device_type, _device_index);

        }

        public DeviceSpec(DeviceSpec other)
        {
            _job = other._job;
            _replica = other._replica;
            _task = other._task;
            _device_type = other._device_type;
            _device_index = other._device_index;
            _as_string = other._as_string;
        }

        protected DeviceSpec(Components com)
        {
            _job = com.Job;
            _replica = com.Replica;
            _task = com.Task;
            _device_type = com.DeviceType;
            _device_index = com.DeviceIndex;
            _as_string = _components_to_string(_job, _replica, _task, _device_type, _device_index);
        }

        public DeviceSpec replace(string job = null, int replica = -1, int task = -1, 
            string device_type = null, int device_index = -1)
        {
            job = job ?? _job;
            replica = replica == -1 ? _replica : replica;
            task = task == -1 ? _task : task;
            device_type = device_type ?? _device_type;
            device_index = device_index == -1 ? _device_index : device_index;
            return new DeviceSpec(job, replica, task, device_type, device_index);
        }

        public static DeviceSpec from_string(string spec)
        {
            var components = _string_to_components(spec);
            return new DeviceSpec(components.Job, components.Replica, components.Task, components.DeviceType, components.DeviceIndex);
        }

        public DeviceSpec make_merged_spec(DeviceSpec dev)
        {
            return new DeviceSpec(_get_combined_properties(dev));
        }

        private Components _get_combined_properties(DeviceSpec dev)
        {
            return new Components(
                dev.Job ?? _job,
                dev.Replica == -1 ? _replica : dev.Replica,
                dev.Task == -1 ? _task : dev.Task,
                dev.DeviceType ?? _device_type,
                dev.DeviceIndex == -1 ? _device_index : dev.DeviceIndex
                );
        }

        private static string _components_to_string(string job, int replica, int task, string device_type, int device_index)
        {
            var key = new Components(job, replica, task, device_type, device_index);
            if(_COMPONENTS_TO_STRING_CACHE.TryGetValue(key, out var cache_result))
            {
                return cache_result;
            }

            StringBuilder output = new();
            if(job is not null)
            {
                output.Append($"/job:{job}");
            }
            if(replica != -1)
            {
                output.Append($"/replica:{replica}");
            }
            if(task != -1)
            {
                output.Append($"/task:{task}");
            }
            if (device_type is not null)
            {
                string device_index_string = "*";
                if (device_index != -1)
                {
                    device_index_string = device_index.ToString();
                }
                output.Append($"/device:{device_type}:{device_index_string}");
            }
            var result = output.ToString();
            _COMPONENTS_TO_STRING_CACHE[key] = result;
            return result;
        }

        private static Components _string_to_components(string spec)
        {
            if(_STRING_TO_COMPONENTS_CACHE.TryGetValue(spec, out var cached_result))
            {
                return cached_result;
            }
            var raw_spec = spec;
            var splits = spec.Split('/').Select(x => x.Split(':'));
            var valid_device_types = _get_valid_device_types();
            string job = null, device_type = null;
            int replica = -1, task = -1, device_index = -1;
            foreach (var y in splits)
            {
                var ly = y.Length;
                if (ly > 0)
                {
                    if(ly == 2 && y[0] == "job")
                    {
                        job = y[1];
                    }
                    else if(ly == 2 && y[0] == "replica")
                    {
                        replica = int.Parse(y[1]);
                    }
                    else if(ly == 2 && y[0] == "task")
                    {
                        task = int.Parse(y[1]);
                    }
                    else if((ly == 1 || ly == 2) && valid_device_types.Contains(y[0].ToUpper()))
                    {
                        if (device_type is not null)
                        {
                            throw new ValueError($"Multiple device types are not allowed " +
                                $"while parsing the device spec: {spec}.");
                        }
                        device_type = y[0].ToUpper();
                        if(ly == 2 && y[1] != "*")
                        {
                            device_index = int.Parse(y[1]);
                        }
                    }
                    else if(ly == 3 && y[0] == "device")
                    {
                        if(device_type is not null)
                        {
                            throw new ValueError($"Multiple device types are not allowed " +
                                $"while parsing the device spec: {spec}.");
                        }
                        device_type = y[1];
                        if (y[2] != "*")
                        {
                            device_index = int.Parse(y[2]);
                        }
                    }
                    else if (y[0] != "")
                    {
                        throw new ValueError($"Unknown attribute '{y[0]}' is encountered " +
                            $"while parsing the device spec: {spec}.");
                    }
                }
            }

            var output = new Components(job, replica, task, device_type, device_index);
            _STRING_TO_COMPONENTS_CACHE[raw_spec] = output;
            return output;
        }

        private static HashSet<string> _get_valid_device_types()
        {
            // TODO(Rinne): revise it to calling C API (need customized API).
            return new HashSet<string>(new string[] { "CPU", "GPU" });
        }

        public override string ToString()
        {
            return _as_string;
        }

        protected record class Components(string Job, int Replica, int Task, string DeviceType, int DeviceIndex);
    }
}
