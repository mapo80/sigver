# Repository notes

This project was updated to work with Python 3.10 or newer.

## Converting models to ONNX
Run `python convert_to_onnx.py` to export the provided PyTorch weights to
`models/*.onnx` files. The script requires the `onnx` package.

There are currently no automated tests. After modifying the repository, ensure
that the export script still runs without errors.

## .NET SDK
If the .NET SDK is not available on your machine, you can install it locally
using the `dotnet-install.sh` script included in the repository:

```bash
./dotnet-install.sh --channel 8.0
```
This downloads a recent SDK to `~/.dotnet` which is then used by the C#
projects.
