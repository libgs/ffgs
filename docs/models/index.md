# Models

The models ffgs ships. Each has a page with its source, licenses, zoo entries,
usage notes and results against the official code.

| model | zoo names | poses | input views | input size | license (code / weights) | page |
|---|---|---|---|---|---|---|
| TokenGS | `tokengs/*` | required | 2 / 4 / 6 | 256×256, 256×448 | Apache-2.0 / non-commercial | [tokengs.md](tokengs.md) |
| AnySplat | `anysplat/default` | optional (predicts the cameras) | any | 448×448 | non-commercial / non-commercial | [anysplat.md](anysplat.md) |

`ffgs.zoo.list_models()` lists every zoo name; each zoo entry also records its
source, pinned revision and license.

To add a model to this list, see [Adding a model](../../README.md#adding-a-model).
