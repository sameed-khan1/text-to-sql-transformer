import json
import matplotlib.pyplot as plt


with open(
    "results.json",
    encoding="utf-8"
) as f:
    results = json.load(f)


epochs = range(
    1,
    results["epochs"] + 1
)

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    results["train_losses"],
    label="Training Loss"
)

plt.plot(
    epochs,
    results["validation_losses"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "training_validation_loss.png",
    dpi=300
)

plt.show()

print("Saved: training_validation_loss.png")