# Synthetic pretraining → private fine-tuning (historical; not included)

**What it was.** Earlier in the project, a Swin-T was pretrained on the 3,000 CC-NET images without DP ("E3.1 / E4") and
then fine-tuned on real images with DP-SGD ("E5"), alongside a Real-only DP family ("E2").

**Why these runs are not included:**
- **Different protocol from the clean experiments:**
  - single linear head;
  - an image-level 20% validation carve of the training set, which the CC-NET generator had (inferred) seen;
  - seeds 42/123/456.
- **Single pretrained model.** All E5 runs fine-tune one pretrained model (seed 42), so their seed spread measures
  fine-tuning only.
- **Overlapping result directories.** Several generations of result directories with different code versions coexist
  for this family.

**What the repository contains for it.** The code paths still exist in `train_classifier.py` (the default `linear`
head; `--pretrained_checkpoint`). However, the repository provides **no configuration or results** for these runs,
and they must not be mixed with the clean reference-head results. Releasing them would need a separate, documented
protocol and the author's decision.
