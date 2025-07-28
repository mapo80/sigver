using SigVerSdk;

if (args.Length < 2)
{
    Console.WriteLine("Usage: FeatureDist <ref> <cand>");
    return;
}

var model = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "models", "signet.onnx"));
Console.Error.WriteLine($"model={model}");
using var verifier = new SigVerifier(model);
var refPath = args[0];
var candPath = args[1];
var refFeat = verifier.ExtractFeatures(refPath);
var candFeat = verifier.ExtractFeatures(candPath);
double sum = 0.0;
for (int i = 0; i < refFeat.Length; i++)
{
    double diff = refFeat[i] - candFeat[i];
    sum += diff * diff;
}
var dist = Math.Sqrt(sum);
Console.WriteLine(dist.ToString(System.Globalization.CultureInfo.InvariantCulture));
