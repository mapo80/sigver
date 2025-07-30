using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Xunit;

namespace SigVerSdk.Tests;

public class MetricsTests
{
    [Fact]
    public void ComputeMetricsForDataset()
    {
        var root = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../"));
        var dataDir = Path.Combine(root, "data");
        var modelPath = Path.Combine(root, "models", "signet.onnx");
        using var verifier = new SigVerSdk.SigVerifier(modelPath);

        var cache = new Dictionary<string, float[]>();
        float[] GetFeatures(string path)
        {
            if (!cache.TryGetValue(path, out var feat))
            {
                feat = verifier.ExtractFeatures(path);
                SigVerSdk.SigVerifier.Normalize(feat);
                cache[path] = feat;
            }
            return feat;
        }

        double Distance(string a, string b)
        {
            var f1 = GetFeatures(a);
            var f2 = GetFeatures(b);
            return SigVerSdk.SigVerifier.CosineDistance(f1, f2);
        }

        List<double> genuineDists = new();
        List<double> forgeryDists = new();
        foreach (var dir in Directory.GetDirectories(dataDir))
        {
            var name = Path.GetFileName(dir);
            if (name.EndsWith("_forg"))
                continue;
            var genuineFiles = Directory.GetFiles(dir);
            var forgeryDir = Path.Combine(dataDir, name + "_forg");
            var forgeryFiles = Directory.Exists(forgeryDir) ? Directory.GetFiles(forgeryDir) : Array.Empty<string>();

            for (int i = 0; i < genuineFiles.Length; i++)
            {
                for (int j = i + 1; j < genuineFiles.Length; j++)
                {
                    genuineDists.Add(Distance(genuineFiles[i], genuineFiles[j]));
                }
            }

            foreach (var g in genuineFiles)
            {
                foreach (var f in forgeryFiles)
                {
                    forgeryDists.Add(Distance(g, f));
                }
            }
        }

        Assert.NotEmpty(genuineDists);
        Assert.NotEmpty(forgeryDists);

        static (double mean,double std,double min,double p25,double median,double p75,double max) Stats(List<double> values)
        {
            var arr = values.ToArray();
            Array.Sort(arr);
            double mean = arr.Average();
            double std = Math.Sqrt(arr.Select(v => (v - mean) * (v - mean)).Average());
            double min = arr.First();
            double max = arr.Last();
            double p25 = arr[(int)(0.25 * (arr.Length - 1))];
            double median = arr[(int)(0.5 * (arr.Length - 1))];
            double p75 = arr[(int)(0.75 * (arr.Length - 1))];
            return (mean, std, min, p25, median, p75, max);
        }

        var gStats = Stats(genuineDists);
        var fStats = Stats(forgeryDists);

        double threshold = 0.35;
        int tp = genuineDists.Count(d => d <= threshold);
        int fn = genuineDists.Count - tp;
        int fp = forgeryDists.Count(d => d <= threshold);
        int tn = forgeryDists.Count - fp;

        Console.WriteLine($"Class\tCount\tMean\tStdDev\tMin\tP25\tMedian\tP75\tMax");
        Console.WriteLine($"Genuine\t{genuineDists.Count}\t{gStats.mean:F4}\t{gStats.std:F4}\t{gStats.min:F4}\t{gStats.p25:F4}\t{gStats.median:F4}\t{gStats.p75:F4}\t{gStats.max:F4}");
        Console.WriteLine($"Forgery\t{forgeryDists.Count}\t{fStats.mean:F4}\t{fStats.std:F4}\t{fStats.min:F4}\t{fStats.p25:F4}\t{fStats.median:F4}\t{fStats.p75:F4}\t{fStats.max:F4}");
        Console.WriteLine($"TP={tp} FN={fn} FP={fp} TN={tn}");

        double accuracy = (double)(tp + tn) / (tp + tn + fp + fn);
        double precision = tp / (double)(tp + fp);
        double recall = tp / (double)(tp + fn);
        double f1 = 2 * precision * recall / (precision + recall);
        Console.WriteLine($"Accuracy={accuracy:F4} Precision={precision:F4} Recall={recall:F4} F1={f1:F4}");

        var all = genuineDists.Select(d => (dist: d, label: 1)).Concat(forgeryDists.Select(d => (dist: d, label: 0)))
            .OrderBy(t => t.dist).ToArray();
        int P = genuineDists.Count;
        int N = forgeryDists.Count;
        double prevFpr = 0, prevTpr = 0, auc = 0;
        double eerDiff = double.MaxValue; double eer = 0; double eerThr = threshold;
        int tpCount = 0, fpCount = 0;
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
                eerDiff = diff; eer = (fprVal + fnrVal) / 2; eerThr = item.dist;
            }
        }
        auc = Math.Clamp(auc, 0, 1);
        Console.WriteLine($"AUC={auc:F4} EER={eer:F4} thr@EER={eerThr:F4}");

        double bhatta = 0.25 * Math.Log(0.25 * (gStats.std*gStats.std/(fStats.std*fStats.std) + fStats.std*fStats.std/(gStats.std*gStats.std) + 2))
                         + 0.25 * (gStats.mean - fStats.mean) * (gStats.mean - fStats.mean) / (gStats.std*gStats.std + fStats.std*fStats.std);
        Console.WriteLine($"Bhattacharyya={bhatta:F4}");

        Assert.InRange(accuracy, 0.0, 1.0);
        Assert.InRange(precision, 0.0, 1.0);
        Assert.InRange(recall, 0.0, 1.0);
        Assert.InRange(f1, 0.0, 1.0);
    }
}
