using SigVerSdk;

if (args.Length < 2)
{
    Console.WriteLine("Usage: PreprocessDump <outputDir> <image1> [image2 ...]");
    return;
}

var model = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "models", "signet.onnx"));
using var verifier = new SigVerifier(model);
var outDir = args[0];
Directory.CreateDirectory(outDir);
for (int i = 1; i < args.Length; i++)
{
    var src = args[i];
    var name = Path.GetFileNameWithoutExtension(src) + "_dotnet.png";
    var dst = Path.Combine(outDir, name);
    verifier.SavePreprocessed(src, dst);
    Console.WriteLine($"Saved {dst}");
}
