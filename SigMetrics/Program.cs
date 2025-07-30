using SigVerSdk;
using System.Globalization;
using System.Linq;
using System.Collections.Generic;

// Command line: SigMetrics <dataDir> [threshold] [w|grid]

if (args.Length < 1)
{
    Console.WriteLine("Usage: SigMetrics <dataDir> [threshold] [w|grid]");
    return;
}

string dataDir = args[0];
float threshold = args.Length >= 2 ? float.Parse(args[1], CultureInfo.InvariantCulture) : 0.35f;
bool grid = args.Length >= 3 && args[2].Equals("grid", StringComparison.OrdinalIgnoreCase);
float w = args.Length >= 3 && !grid ? float.Parse(args[2], CultureInfo.InvariantCulture) : 0.5f;

var basePath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", ".."));
var modelPath = Path.Combine(basePath, "models", "signet.onnx");
var modelPath2 = Path.Combine(basePath, "models", "signet_f_lambda_0.95.onnx");
using var verifier = new SigVerifier(modelPath);
using var verifier2 = new SigVerifier(modelPath2);

var cache1 = new Dictionary<string,double[]>();
var cache2 = new Dictionary<string,double[]>();

List<double> d1Gen = new();
List<double> d1For = new();
List<double> d2Gen = new();
List<double> d2For = new();

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
            DistancePair(genuineFiles[i], genuineFiles[j], out double d1, out double d2);
            d1Gen.Add(d1);
            d2Gen.Add(d2);
        }
    }

    foreach (var g in genuineFiles)
    {
        foreach (var f in forgeryFiles)
        {
            DistancePair(g, f, out double d1, out double d2);
            d1For.Add(d1);
            d2For.Add(d2);
        }
    }
}

// compute fused scores
List<double> avgGen = d1Gen.Zip(d2Gen, (a,b) => (a+b)/2).ToList();
List<double> avgFor = d1For.Zip(d2For, (a,b) => (a+b)/2).ToList();
List<double> minGen = d1Gen.Zip(d2Gen, Math.Min).ToList();
List<double> minFor = d1For.Zip(d2For, Math.Min).ToList();
List<double> maxGen = d1Gen.Zip(d2Gen, Math.Max).ToList();
List<double> maxFor = d1For.Zip(d2For, Math.Max).ToList();

Evaluate("SigNet", d1Gen, d1For);
Evaluate("SigNet-F", d2Gen, d2For);
Evaluate("avg", avgGen, avgFor);

if (grid)
{
    for (int i=0;i<=10;i++)
    {
        float ww = i/10f;
        var g = d1Gen.Zip(d2Gen, (a,b)=> ww*a + (1-ww)*b).ToList();
        var f = d1For.Zip(d2For, (a,b)=> ww*a + (1-ww)*b).ToList();
        Evaluate($"w={ww:F1}", g, f);
    }
}
else
{
    var wGen = d1Gen.Zip(d2Gen, (a,b)=> w*a + (1-w)*b).ToList();
    var wFor = d1For.Zip(d2For, (a,b)=> w*a + (1-w)*b).ToList();
    Evaluate($"w={w:F1}", wGen, wFor);
}

Evaluate("min", minGen, minFor);
Evaluate("max", maxGen, maxFor);

double[] Get(SigVerifier v, Dictionary<string,double[]> cache, string path)
{
    if (!cache.TryGetValue(path, out var vec))
    {
        var f = v.ExtractFeatures(path);
        SigVerifier.Normalize(f);
        vec = f.Select(x => (double)x).ToArray();
        cache[path] = vec;
    }
    return vec;
}

void DistancePair(string a, string b, out double d1, out double d2)
{
    var f1_1 = Get(verifier, cache1, a); var f1_2 = Get(verifier, cache1, b);
    d1 = SigVerifier.CosineDistance(f1_1.Select(v=>(float)v).ToArray(), f1_2.Select(v=>(float)v).ToArray());
    var f2_1 = Get(verifier2, cache2, a); var f2_2 = Get(verifier2, cache2, b);
    d2 = SigVerifier.CosineDistance(f2_1.Select(v=>(float)v).ToArray(), f2_2.Select(v=>(float)v).ToArray());
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

void Evaluate(string name, List<double> gen, List<double> forg)
{
    var gStats = Stats(gen);
    var fStats = Stats(forg);
    int tp = gen.Count(d => d <= threshold);
    int fn = gen.Count - tp;
    int fp = forg.Count(d => d <= threshold);
    int tn = forg.Count - fp;
    Console.WriteLine($"{name}");
    Console.WriteLine($"Class\tCount\tMean\tStdDev\tMin\tP25\tMedian\tP75\tMax");
    Console.WriteLine($"Genuine\t{gen.Count}\t{gStats.mean:F4}\t{gStats.std:F4}\t{gStats.min:F4}\t{gStats.p25:F4}\t{gStats.median:F4}\t{gStats.p75:F4}\t{gStats.max:F4}");
    Console.WriteLine($"Forgery\t{forg.Count}\t{fStats.mean:F4}\t{fStats.std:F4}\t{fStats.min:F4}\t{fStats.p25:F4}\t{fStats.median:F4}\t{fStats.p75:F4}\t{fStats.max:F4}");
    Console.WriteLine($"TP={tp} FN={fn} FP={fp} TN={tn}");

    double accuracy = (double)(tp + tn) / (tp + tn + fp + fn);
    double precision = tp + fp == 0 ? 0 : tp / (double)(tp + fp);
    double recall = tp + fn == 0 ? 0 : tp / (double)(tp + fn);
    double f1 = precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);
    Console.WriteLine($"Accuracy={accuracy:F4} Precision={precision:F4} Recall={recall:F4} F1={f1:F4}");

    var all = gen.Select(d => (dist: d, label: 1)).Concat(forg.Select(d => (dist: d, label: 0)))
        .OrderBy(t => t.dist).ToArray();
    int P = gen.Count;
    int N = forg.Count;
    double prevFpr = 0, prevTpr = 0, auc = 0;
    double eerDiff = double.MaxValue; double eerThr = threshold; double eer = 0;
    int tpCount = 0, fpCount = 0;
    double thr1 = double.NaN, thr5 = double.NaN;
    foreach (var item in all)
    {
        if (item.label == 1) tpCount++; else fpCount++;
        double tprVal = tpCount / (double)P;
        double fprVal = fpCount / (double)N;
        auc += (fprVal - prevFpr) * (tprVal + prevTpr) / 2.0;
        prevFpr = fprVal; prevTpr = tprVal;
        double fnrVal = 1 - tprVal;
        double diff = Math.Abs(fprVal - fnrVal);
        if (diff < eerDiff)
        {
            eerDiff = diff; eerThr = item.dist; eer = (fprVal + fnrVal)/2;
        }
        if (double.IsNaN(thr1) && fprVal <= 0.01) thr1 = item.dist;
        if (double.IsNaN(thr5) && fprVal <= 0.05) thr5 = item.dist;
    }
    auc = Math.Clamp(auc, 0, 1);

    int Conf(double thr, out double prec)
    {
        int tp2 = gen.Count(d => d <= thr);
        int fn2 = gen.Count - tp2;
        int fp2 = forg.Count(d => d <= thr);
        int tn2 = forg.Count - fp2;
        prec = tp2 + fp2 == 0 ? 0 : tp2 / (double)(tp2 + fp2);
        return tp2 + tn2;
    }

    double prec1=0, prec5=0;
    if (!double.IsNaN(thr1)) Conf(thr1, out prec1);
    if (!double.IsNaN(thr5)) Conf(thr5, out prec5);

    double bhatta = 0.25 * Math.Log(0.25 * (gStats.std*gStats.std/(fStats.std*fStats.std) + fStats.std*fStats.std/(gStats.std*gStats.std) + 2))
                     + 0.25 * (gStats.mean - fStats.mean) * (gStats.mean - fStats.mean) / (gStats.std*gStats.std + fStats.std*fStats.std);

    var pr = LearnLogistic(all);
    double logloss = 0; double ece = 0; int bins = 10; double m = all.Length;
    for (int i=0;i<all.Length;i++)
        logloss += - (all[i].label * Math.Log(pr[i]) + (1 - all[i].label) * Math.Log(1 - pr[i]));
    logloss /= m;
    for (int b=0;b<bins;b++)
    {
        double lo=(double)b/bins, hi=(double)(b+1)/bins;
        var subset = Enumerable.Range(0,all.Length).Where(i => pr[i]>=lo && pr[i]<hi).ToArray();
        if (subset.Length==0) continue;
        double avgConf = subset.Average(i => pr[i]);
        double avgAcc = subset.Average(i => all[i].label);
        ece += subset.Length/m * Math.Abs(avgAcc - avgConf);
    }

    Console.WriteLine($"AUC={auc:F4} EER={eer:F4} thr@EER={eerThr:F4}");
    Console.WriteLine($"Precision@FPR1%={prec1:F4} Precision@FPR5%={prec5:F4}");
    Console.WriteLine($"Bhattacharyya={bhatta:F4} LogLoss={logloss:F4} ECE={ece:F4}\n");
}

static double[] LearnLogistic((double dist,int label)[] data)
{
    double a=0,b=0,lr=0.1; int steps=1000; int n=data.Length;
    for(int it=0;it<steps;it++)
    {
        double ga=0,gb=0;
        foreach(var d in data)
        {
            double p=1/(1+Math.Exp(-(a*d.dist+b)));
            double e=p-d.label;
            ga+=e*d.dist; gb+=e;
        }
        ga/=n; gb/=n; a-=lr*ga; b-=lr*gb;
    }
    return data.Select(d=>1/(1+Math.Exp(-(a*d.dist+b)))).ToArray();
}
