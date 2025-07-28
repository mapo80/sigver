using System;
using SigVerSdk;

if (args.Length < 4)
{
    Console.WriteLine("Usage: SigCompare <model> <img1> <img2> <threshold>");
    return 1;
}

string modelPath = args[0];
string img1 = args[1];
string img2 = args[2];
if (!float.TryParse(args[3], out float threshold))
{
    Console.WriteLine("Invalid threshold");
    return 1;
}

using var verifier = new SigVerifier(modelPath);
var f1 = verifier.ExtractFeatures(img1);
var f2 = verifier.ExtractFeatures(img2);
SigVerifier.Normalize(f1);
SigVerifier.Normalize(f2);
double distance = SigVerifier.CosineDistance(f1, f2);
bool isForged = distance > threshold;
Console.WriteLine($"{distance:F6} {isForged}");
return 0;
