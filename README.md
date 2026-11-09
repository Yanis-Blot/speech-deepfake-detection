# Deepfake detection

In this work, we are testing and comparing few ML approaches for deepfake recognition. Among them: MLP, CNN, RNN, LSTM, GRU, Transformers, hybrid pipelines, and pre-trained models.

Our goal is to make a comprehensive comparison, and to show, and more importantly, explain why an architecture performs better than another for this specific task.


# Audio Deepfake Detection — Project Process

Consolidated pipeline combining the 3-model plan (MLP baseline, CNN, wav2vec2) with the
agreed adjustments: RNN coverage, class imbalance handling, and correct use of
`system_id` / `speaker_id`.

## 1. Load the data
Load `Bisher/ASVspoof_2019_LA` (train/validation/test, already split with disjoint speakers).

## 2. Sanity checks (before any training)
- Verify no `speaker_id` appears in more than one split (data leakage check)
- Compute bonafide/spoof proportions per split → confirm the imbalance 

## 3. Audio preprocessing (common to all models except wav2vec2)
- Resample to 16 kHz
- Pad/crop to a fixed duration (e.g. 4 sec)
- Amplitude normalization
