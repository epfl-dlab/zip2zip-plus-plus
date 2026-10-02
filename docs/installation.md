# Installation

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Rust toolchain (for building `zip2zip-compression`)

## Setup

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/Saibo-creator/zip2zip-plus-plus.git
cd zip2zip-plus-plus
```

Install everything (creates venv, builds zip2zip-compression from Rust, installs torchtitan from submodule):

```bash
uv sync
```

For data preprocessing (tokenization), include the optional dependencies:

```bash
uv sync --extra data
```
