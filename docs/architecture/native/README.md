# Native Runtime

Use this subtree for the native C++/CUDA train and serve contracts.

## Read This Section When

- You need the product runtime boundary after Python removal.
- You need the native artifact format.
- You need CUDA kernel ownership rules.
- You need train, export, and serve acceptance gates.

## Child Index

- [strategy.md](strategy.md): rewrite boundary and migration order
- [artifact.md](artifact.md): native checkpoint and weight files
- [runtime.md](runtime.md): HTTP server and inference behavior
- [training.md](training.md): native trainer ownership and data flow
- [kernels.md](kernels.md): CUDA library and custom-kernel rules
