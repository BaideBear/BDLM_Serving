# DiffusionGemma: certified Q256 native step

Three actual runs per frozen request, same seed, same n164 H0, physical
B=Q=256. This is a reference endpoint for engineering the unified request:
it matches native, not an improvement over native.

| Arm | Correct | Macro TPF | Pooled token efficiency | Generation F | Decoder Q | Generation encoder F |
|---|---:|---:|---:|---:|---:|---:|
| Upstream `model.generate` | 85/90 | 25.510 | 9.31% | 1,397 | 336,896 | 81 |
| Native-argmax `request.step(256)` | 85/90 | 25.510 | 9.31% | 1,397 | 336,896 | 81 |
| Extra paid verify | 85/90 | 21.976 | 8.26% | 1,565 | 379,904 | 81 |

Both the native arm and the step delivered the exact same IDs on all 90
requests, including the 59 multiblock requests; all 81 encoder inputs and
171 hit argmaxes matched. The extra verify changed 18 block deliveries from
its own hit draft, including seven nonterminal encoder inputs. Same overall
score need not mean same output. An extra verify is not harmless and is not
part of this native endpoint. Do not transfer the result to a partial-block
or two-block call.

[Annotated comparison](frozen90.png) · [Plot script](plot.py) · [Original 30 / new 60 / all 90](cohorts.csv) · [Per-request metrics and hashes](pairs.csv)
· [Implementation, receipt hashes and hardware](../../../../../../docs/gemma-b-native-argmax-gpu-qualification.md).
TPF is `mean(N_i/F_i)` where F includes generation encoders. Token efficiency
is `sum(N_i)/sum(D_Q_i)`; neither prompt prefill nor encoder Q belongs in
the denominator. All three arms bill every real decoder and generation
encoder call.
