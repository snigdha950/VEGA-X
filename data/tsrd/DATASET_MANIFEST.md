# TSRD replay manifest

All four files have a `data` matrix with five float32 columns:
`ToA, Frequency, PulseWidth, AoA, Amplitude`. The loader uses only ToA and
Frequency to construct hidden replay truth.

| File | Rows | SHA-256 | Status |
|---|---:|---|---|
| `config_0.h5` | 648,034 | `d25a4189f287ca044f46718baea7471eb1f9ab766b7715c738bb8edf8ac1e474` | Supplied file; byte-identical to the separately uploaded `config_0(2).h5` |
| `config_1.h5` | 1,234,196 | `6d182dac7803bef76aafede9752f19dc70e8422d287bf33f2701b78947e4bfa9` | Retained from prior complete archive |
| `config_10.h5` | 463,123 | `27c6d49e2a1c45620228161dc5df015fae88c2ac164d7684e2057b1055b03251` | Retained from prior complete archive |
| `config_103.h5` | 3,357 | `ab25d733c166162ff921837d09f7e4d14399443eb42643ce5b029e26b7f90f5f` | Retained from prior complete archive |

Validated checks:

- non-empty two-dimensional `data` dataset;
- feature-name count matches the data-column count;
- unique `ToA` and `Frequency` fields are present;
- no labels or transmitter metadata are passed into a scheduling policy.
