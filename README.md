# RAFTS: Receiver-Aware Frequency-Time Thompson Scheduler
## SIH26055 — Smart Scan Strategy for Electronic Warfare

RAFTS is a software prototype for scheduling a limited number of Electronic
Support receivers over a wider RF spectrum without reliable prior emitter
intelligence.

### Core design
1. Discounted Thompson Sampling learns online from receiver HIT/MISS feedback.
2. A confidence-gated periodic specialist contributes bounded timing evidence
   only after sufficient receiver-observed hits.
3. A deadline-aware Max-Age guard prevents indefinite band starvation. It
   reserves retune/observation capacity before a deadline and resets age only
   after an actual completed observation.
4. Joint receiver-band assignment accounts for physical retune dead-time.
5. Hidden RF truth is isolated inside the environment/receiver boundary.
6. TSRD is used as externally generated radar replay, not as privileged
   scheduler knowledge.

### TSRD mapping
TSRD pulse-level PDWs are aggregated into a frequency x time replay grid:
- ToA -> RAFTS time slot
- Centre Frequency -> RAFTS frequency band
- one or more pulses in a cell -> hidden transmission state by default

PulseWidth, AoA, Amplitude, emitter labels and transmitter metadata are NOT
fed to the online scheduler in this version.

This aggregation is deliberate: individual radar pulses are not treated as
independent interception episodes.

### Metrics
The prototype reports:
- Interception Ratio
- Average Intercept Rate
- Conditional Mean / P95 Time To Intercept
- Average Intercept-Time Error under the frozen onset-to-first-detection definition
- Empirical Pd / Pfa
- Correct observation fraction
- Average reward per observation
- Retunes and retune dead-time
- Max-Age coverage violations

Sensitivity is a configurable simulation parameter. It is NOT presented as a
calibrated hardware measurement.

### Run
Install:
    python -m pip install -r requirements.txt

Copy TSRD files into data/tsrd/, then:
    python audit.py
    python tests_edge_cases.py
    python dataset_replay.py --file data/tsrd/config_103.h5
    python dataset_replay.py --file data/tsrd/config_0.h5
    python validate_tsrd.py data/tsrd/config_0.h5 data/tsrd/config_1.h5 data/tsrd/config_10.h5 --seeds 10

### Fair comparison
Every policy receives the same hidden replay and same receiver physics.
RAFTS, DTS, Sequential Sweep and Random Scan are compared without exposing
future TSRD truth to any scheduler.

### Scope statement
This is a research/software prototype. It does not claim calibrated real-world
receiver sensitivity or deployment on operational EW hardware. SDR/hardware
integration is future validation.
