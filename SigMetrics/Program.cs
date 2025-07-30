using SigVerSdk;
using System.Globalization;

if (args.Length < 1)
{
    Console.WriteLine("Usage: SigMetrics <dataDir> [threshold]");
    return;
}

string dataDir = args[0];
float threshold = args.Length >= 2 ? float.Parse(args[1], CultureInfo.InvariantCulture) : 0.35f;

var modelPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "models", "signet.onnx"));
using var verifier = new SigVerifier(modelPath);

List<double> genuineDists = new();
List<double> forgeryDists = new();

foreach (var dir in Directory.GetDirectories(dataDir))
{
    string name = Path.GetFileName(dir);
    if (name.EndsWith("_forg"))
        continue;
    string genuDir = dir;
    string forgDir = Path.Combine(dataDir, name + "_forg");
    var genuineFiles = Directory.GetFiles(genuDir);
    var forgeryFiles = Directory.Exists(forgDir) ? Directory.GetFiles(forgDir) : Array.Empty<string>();

    for (int i = 0; i < genuineFiles.Length; i++)
    {
        for (int j = i + 1; j < genuineFiles.Length; j++)
        {
            genuineDists.Add(Distance(verifier, genuineFiles[i], genuineFiles[j]));
        }
    }

    foreach (var g in genuineFiles)
    {
        foreach (var f in forgeryFiles)
        {
            forgeryDists.Add(Distance(verifier, g, f));
        }
    }
}

static double Distance(SigVerifier v, string a, string b)
{
    var f1 = v.ExtractFeatures(a);
    var f2 = v.ExtractFeatures(b);
    SigVerifier.Normalize(f1);
    SigVerifier.Normalize(f2);
    return SigVerifier.CosineDistance(f1, f2);
}

static (double mean,double std,double min,double p25,double median,double p75,double max) Stats(List<double> values)
{
    var arr = values.ToArray();
    Array.Sort(arr);
    double mean = arr.Average();
    double std = Math.Sqrt(arr.Select(v => (v - mean)*(v - mean)).Average());
    double min = arr.First();
    double max = arr.Last();
    double p25 = arr[(int)(0.25*(arr.Length-1))];
    double median = arr[(int)(0.5*(arr.Length-1))];
    double p75 = arr[(int)(0.75*(arr.Length-1))];
    return (mean,std,min,p25,median,p75,max);
}

var gStats = Stats(genuineDists);
var fStats = Stats(forgeryDists);
int tp = genuineDists.Count(d => d <= threshold);
int fn = genuineDists.Count - tp;
int fp = forgeryDists.Count(d => d <= threshold);
int tn = forgeryDists.Count - fp;
Console.WriteLine($"Class\tCount\tMean\tStdDev\tMin\tP25\tMedian\tP75\tMax");
Console.WriteLine($"Genuine\t{genuineDists.Count}\t{gStats.mean:F4}\t{gStats.std:F4}\t{gStats.min:F4}\t{gStats.p25:F4}\t{gStats.median:F4}\t{gStats.p75:F4}\t{gStats.max:F4}");
Console.WriteLine($"Forgery\t{forgeryDists.Count}\t{fStats.mean:F4}\t{fStats.std:F4}\t{fStats.min:F4}\t{fStats.p25:F4}\t{fStats.median:F4}\t{fStats.p75:F4}\t{fStats.max:F4}");
Console.WriteLine($"TP={tp} FN={fn} FP={fp} TN={tn}");
