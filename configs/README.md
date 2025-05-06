# Chromatin Simulation Configurations

This directory contains JSON configuration files for running the Active Cahn-Hilliard simulation with different noise levels for heterochromatin and euchromatin regions.

## Available configurations

| Configuration | Euchromatin Noise | Heterochromatin Noise | Description |
|---------------|------------------|----------------------|-------------|
| `low_eu_low_het.json` | 0.05 | 0.05 | Low noise in both regions |
| `low_eu_med_het.json` | 0.05 | 0.3 | Low euchromatin noise, medium heterochromatin noise |
| `low_eu_high_het.json` | 0.05 | 0.8 | Low euchromatin noise, high heterochromatin noise |
| `med_eu_low_het.json` | 0.3 | 0.05 | Medium euchromatin noise, low heterochromatin noise |
| `high_eu_low_het.json` | 0.8 | 0.05 | High euchromatin noise, low heterochromatin noise |
| `high_eu_high_het.json` | 0.8 | 0.8 | High noise in both regions |

## Key parameters

- `noise_amp`: Base noise amplitude (0.5)
- `euchromatin_noise_factor`: Factor to scale noise in euchromatin regions
- `heterochromatin_noise_factor`: Factor to scale noise in heterochromatin regions
- `allow_overlap`: Whether heterochromatin circles can overlap (true)
- `overlap_strength`: How much density is added when circles overlap (1.0 = linear addition)

## Running the simulations

To run a single simulation:
```bash
./run_single_sim.sh configs/low_eu_low_het.json 42
```

To run all simulations:
```bash
./run_all_sims.sh
```

Each simulation will create its own output directory with the timestamp in its name. 