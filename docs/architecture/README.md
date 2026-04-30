# Architecture

Use this subtree for the agent, memory, model, native runtime, and training
contracts.

## Read This Section When

- You need the multi-turn agent loop.
- You need structured tool and event contracts.
- You need scratch model training, serving, memory, or agent behavior.
- You need exact ownership boundaries between web runtime, model API, and train stack.
- You need the native C++/CUDA product path.

## Child Index

- [agent/README.md](agent/README.md): plan-act-observe-revise loop and schemas
- [memory/README.md](memory/README.md): transcript, summary, durable memory, and retrieval
- [model/README.md](model/README.md): scratch model selection and serving defaults
- [native/README.md](native/README.md): C++/CUDA train and serve contracts
- [training/README.md](training/README.md): tokenizer, corpus, and scratch training
- [runtime/README.md](runtime/README.md): web, model client, and storage behavior
