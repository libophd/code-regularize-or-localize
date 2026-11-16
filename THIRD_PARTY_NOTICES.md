# Third-party materials and provenance

The original project code and documentation are released under the MIT
License in `LICENSE`. Third-party materials retain their existing terms;
the project license does not relicense them or override the paper's
publication terms.

## SIGReg and the LeWorldModel reference implementation

`src/jepa_lm/training/sigreg.py` implements the Sketched Isotropic Gaussian
Regularizer (SIGReg) introduced by LeJEPA and used by LeWorldModel. It was
re-derived from the appendices of those papers and checked line by line
against the reference implementation in the LeWorldModel repository
(`class SIGReg` in <https://github.com/lucas-maes/le-wm/blob/main/module.py>).
It keeps that reference's quadrature (17 knots on [0, 3], trapezoid weights,
Gaussian window), and two of its lines, the window and the Epps-Pulley error
term, are nearly identical to lines of the reference. Its per-position variant
matches the semantics of the same class. The reduction of the prediction loss
in `src/jepa_lm/training/losses.py` and the layer order of the projector in
`src/jepa_lm/trunk_heads/projector.py` follow conventions of that repository
(`train.py` and `module.py`).

The LeWorldModel repository is distributed under the MIT License. Its
license file, as published at
<https://github.com/lucas-maes/le-wm/blob/main/LICENSE> on 2026-11-15, is
reproduced in full:

```text
MIT License

Copyright (c) 2026 Lucas Maes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

A class with the same content also appears in `MINIMAL.md` of the official
LeJEPA repository (<https://github.com/rbalestr-lab/lejepa>), which is
distributed under CC BY-NC 4.0. The reference used for this project was the
MIT-licensed LeWorldModel file named above.

## Other third-party materials

- The Llama implementation uses Hugging Face Transformers. Transformers,
  PyTorch, and other installed dependencies retain their own licenses.
- Dataset, tokenizer, and benchmark references in the code and in the result
  files do not confer redistribution rights. The export contains no FineWeb
  text, tokenizer vocabulary, pretrained weights, or model checkpoint
  binaries.

The curated result files are included to support the paper; keep the source
attributions and upstream notices with them.
