# Numerical providers

Auto selects Accelerate on macOS when compiled in, otherwise Portable. Explicit
unavailable providers fail at preparation. QA also supports independent FFT/math
substitutions. See [qualification](QUALIFICATION.md) for current measurements and
[historical results](baselines/historical-core.json) for the prior campaign.
