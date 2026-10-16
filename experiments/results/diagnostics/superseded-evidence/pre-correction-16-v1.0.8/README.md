# Superseded evidence: pre-Correction 16 diagnostic

These files preserve the previous generator and its outputs byte for byte.
They are retained for audit, not as valid engine-observation evidence.
The generator drew from NumPy, clipped the draws, and shifted them toward
preset means. It did not call NetworkImpairmentEngine._apply_latency().
The old CSV therefore does not contain raw observations from that engine.
The previous summary also hard-coded empirical cohort drop rates and made
unsupported claims about dropout-theory agreement and engine overhead.

The adjacent evidence_manifest.json records original paths and SHA-256 hashes.
The active corrected outputs are in the diagnostics directory above this archive.
