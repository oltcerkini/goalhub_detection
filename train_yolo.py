"""Fine-tune YOLO26m on Soccana football dataset."""
import multiprocessing

CHECKPOINT = "soccana_dataset/runs/detect/goalhub_finetune/yolo26m_soccana/weights/last.pt"

# Required for Windows multiprocessing support
if __name__ == "__main__":
    multiprocessing.freeze_support()
    from ultralytics import YOLO

    model = YOLO(CHECKPOINT)
    results = model.train(resume=True)

    print("Training complete!")
    print(results)
