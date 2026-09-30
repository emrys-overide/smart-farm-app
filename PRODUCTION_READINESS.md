# Production readiness plan (2026-09-30)

## Assessment: 45% complete before this branch
Flask backend and a substantial single-page interface exist. The committed model path is a Git LFS pointer, while the backend attempts to load it as a PyTorch checkpoint. If the model is absent, inference returns random diagnoses with plausible confidence. Uploads have no size limit or image validation boundary, and the script runs Flask debug mode. No tests or CI are present. The estimate is a rough production-readiness judgment, not feature completion.

## Work completed on this branch
- Make model loading explicitly detect missing or unhydrated LFS checkpoints and report unavailable instead of generating random diagnoses.
- Bound uploads, reject non-image data and malformed images, and return controlled HTTP errors.
- Disable debug mode in the direct run path and add a health endpoint.

## Remaining release gates
1. Supply the authentic trained model through Git LFS or a secure artifact store and verify checksum, tensor shape, class order, and startup inference.
2. Build a representative Kenyan field-image validation set; measure per-class precision, recall, calibration, and out-of-distribution behavior. Review advice with a qualified agronomist.
3. Add automated route and inference tests and GitHub Actions. Pin a tested compatible PyTorch/torchvision dependency set.
4. Remove unused dependencies; review single-file UI for accessibility, privacy, and misleading market/weather data.
5. Add production deployment configuration, HTTPS, logging, rate limiting, monitoring, and rollback; stage-test on actual hardware and network conditions.
