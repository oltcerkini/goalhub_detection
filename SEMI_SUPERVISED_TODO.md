# Semi-Supervised Learning — Round 2

## Goal
Adapt the fine-tuned YOLO26m model to our specific broadcast camera view by training on our own video footage.

## Status
- Training 1 (Soccana dataset): in progress, ~6h remaining (epoch 2/50, started ~2h ago)
- Training 2 (semi-supervised): NOT STARTED — needs user input

## What we need from you
- Which video files to use (e.g., `assets/ulpiana_7sec.mp4` or others)
- How many videos (more = better diversity, ~20-30 min total footage ideal)

## Process (will run automatically)

### Step 1: Extract frames from your video(s)
Extract 1 frame per second from each video → organized into a dataset folder.

### Step 2: Run fine-tuned model on extracted frames
The Soccana model (once trained) detects players, ball, referees on your own footage. Only keep high-confidence detections (conf > 0.7) as "pseudo-labels."

### Step 3: Merge with Soccana + train round 2
Combine your pseudo-labeled frames with the Soccana dataset. Fine-tune another 30-50 epochs. This adapts the model to your specific camera angle, pitch dimensions, and lighting.

## Timeline
- Training 1 completes: ~6h from now
- Frame extraction + pseudo-labeling: ~15 min
- Training 2: ~4-5h

## File locations
- Soccana dataset: `soccana_dataset/V1/`
- User videos: `assets/` directory
- Output models: `goalhub_finetune/yolo26m_soccana/`
