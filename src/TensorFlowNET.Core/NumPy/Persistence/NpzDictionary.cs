using System.IO;
using System.IO.Compression;

namespace Tensorflow.NumPy;

public class NpzDictionary<T> : IDisposable, IReadOnlyDictionary<string, T>, ICollection<T>
        where T : class,
        ICloneable, IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
{
    Stream stream;
    ZipArchive archive;

    bool disposedValue = false;

    Dictionary<string, ZipArchiveEntry> entries;
    Dictionary<string, T> arrays;


    public NpzDictionary(Stream stream)
    {
        this.stream = stream;
        this.archive = new ZipArchive(stream, ZipArchiveMode.Read, leaveOpen: true);

        this.entries = new Dictionary<string, ZipArchiveEntry>();
        foreach (var entry in archive.Entries)
            this.entries[entry.FullName] = entry;

        this.arrays = new Dictionary<string, T>();
    }


    public IEnumerable<string> Keys
    {
        get { return entries.Keys; }
    }


    public IEnumerable<T> Values
    {
        get { return entries.Values.Select(OpenEntry); }
    }

    public int Count
    {
        get { return entries.Count; }
    }


    public object SyncRoot
    {
        get { return ((ICollection)entries).SyncRoot; }
    }


    public bool IsSynchronized
    {
        get { return ((ICollection)entries).IsSynchronized; }
    }

    public bool IsReadOnly
    {
        get { return true; }
    }

    public T this[string key]
    {
        get { return OpenEntry(entries[key]); }
    }

    private T OpenEntry(ZipArchiveEntry entry)
    {
        T array;
        if (arrays.TryGetValue(entry.FullName, out array))
            return array;

        using (Stream s = entry.Open())
        {
            array = Load_Npz(s);
            arrays[entry.FullName] = array;
            return array;
        }
    }

    protected virtual T Load_Npz(Stream s)
    {
        return np.Load<T>(s);
    }

    public bool ContainsKey(string key)
    {
        return entries.ContainsKey(key);
    }

    public bool TryGetValue(string key, out T value)
    {
        value = default(T);
        ZipArchiveEntry entry;
        if (!entries.TryGetValue(key, out entry))
            return false;
        value = OpenEntry(entry);
        return true;
    }

    public IEnumerator<KeyValuePair<string, T>> GetEnumerator()
    {
        foreach (var entry in archive.Entries)
            yield return new KeyValuePair<string, T>(entry.FullName, OpenEntry(entry));
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        foreach (var entry in archive.Entries)
            yield return new KeyValuePair<string, T>(entry.FullName, OpenEntry(entry));
    }

    IEnumerator<T> IEnumerable<T>.GetEnumerator()
    {
        foreach (var entry in archive.Entries)
            yield return OpenEntry(entry);
    }

    public void CopyTo(Array array, int arrayIndex)
    {
        foreach (var v in this)
            array.SetValue(v, arrayIndex++);
    }

    public void CopyTo(T[] array, int arrayIndex)
    {
        foreach (var v in this)
            array.SetValue(v, arrayIndex++);
    }

    public void Add(T item)
    {
        throw new ReadOnlyException();
    }

    public void Clear()
    {
        throw new ReadOnlyException();
    }

    public bool Contains(T item)
    {
        foreach (var v in this)
            if (Object.Equals(v.Value, item))
                return true;
        return false;
    }

    public bool Remove(T item)
    {
        throw new ReadOnlyException();
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!disposedValue)
        {
            if (disposing)
            {
                archive.Dispose();
                stream.Dispose();
            }

            archive = null;
            stream = null;
            entries = null;
            arrays = null;

            disposedValue = true;
        }
    }

    public void Dispose()
    {
        Dispose(true);
    }
}
