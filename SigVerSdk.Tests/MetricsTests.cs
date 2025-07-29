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

        double threshold = 0.35;
        int tp = genuineDists.Count(d => d <= threshold);
        int fn = genuineDists.Count - tp;
        int fp = forgeryDists.Count(d => d <= threshold);
        int tn = forgeryDists.Count - fp;

        double accuracy = (double)(tp + tn) / (tp + tn + fp + fn);
        double precision = tp / (double)(tp + fp);
        double recall = tp / (double)(tp + fn);
        double f1 = 2 * precision * recall / (precision + recall);

        Console.WriteLine($"TP={tp} FN={fn} FP={fp} TN={tn}");
        Console.WriteLine($"Accuracy={accuracy:F4} Precision={precision:F4} Recall={recall:F4} F1={f1:F4}");

        Assert.InRange(accuracy, 0.0, 1.0);
        Assert.InRange(precision, 0.0, 1.0);
        Assert.InRange(recall, 0.0, 1.0);
        Assert.InRange(f1, 0.0, 1.0);
    }
}
