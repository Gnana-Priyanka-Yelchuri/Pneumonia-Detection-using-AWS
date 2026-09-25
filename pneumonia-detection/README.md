# A note on this repo

The original source files for this project (the SageMaker notebook and
related scripts) no longer exist. While cleaning up AWS resources after
the project wrapped up, I deleted the S3(Bucket) to
avoid Supposedly suprise charges — without realizing that also removed my only copy
of the notebook. I'd meant to download it and back it up to local device, but
lost track of that step in the moment and assumed I already had a local
copy. I didn't.

What survived was the project report, which happened to include the full
notebook as inline text and screenshots (code cells, outputs, and all).
I used Claude (Anthropic) to read through the report, cross-check the
text against the notebook screenshots, and reconstruct the original
source as a working notebook (`Pneumonia_Detection_CNN.ipynb`) and script
(`pneumonia_detection.py`).

The logic, parameters, comments, and structure all match what's
documented in the report. One small piece — a short "shutdown cells"
cleanup cell at the very end of the notebook (standard SageMaker
kernel/instance shutdown boilerplate) — wasn't fully recoverable, since
it was cut off in the report and not core project logic.

**Lesson learned:** back up the notebook to Local storage (or push to git) *before*
tearing down any S3 bucket , not after.
