# Phase 1: ingestion and profiling

## Methodology and assumptions

Start with an immutable copy of the UCI archive. The profiler extracts the single workbook, checks the source column names, and records its SHA-256 checksum. It reports row/column counts, per-column missingness and cardinality, exact duplicate count, date range, cancellation invoice count, invalid date count, and suspicious quantity/price counts.

Cancellation invoice numbers are detected using a case-insensitive `C` prefix. That is a flag for later reconciliation, not a full returns ledger. Negative/zero values are surfaced and not silently discarded. Missing `CustomerID` records remain in the source profile because anonymous transactions can still matter for aggregate activity; they cannot enter customer-level analyses without customer identity.

## Validation strategy

The first implementation validates that exactly one workbook exists in the archive and that the workbook schema matches the expected eight source columns. It emits deterministic data-quality counts and a source checksum; `profiled_at_utc` is intentionally run-specific. Later increments will add unit tests for archive/schema handling and explicit cleaning-rule count reconciliation before publishing curated tables.

## Known limitation

This is a source profile, not a cleaning pipeline. In particular, cancellation entries are not automatically paired with original purchases; doing so reliably requires an explicit invoice-line reconciliation rule and audit output in the next data-engineering increment.
