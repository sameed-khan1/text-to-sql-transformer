import json


train_losses = [
    8.1425,
    3.7531,
    3.1174,
    2.8838,
    2.7434,
    2.6340,
    2.5481,
    2.4687,
    2.4010,
    2.3393,
    2.2809,
    2.2294,
    2.1841,
    2.1427,
    2.1069,
    2.0749,
    2.0457,
    2.0202,
    1.9958,
    1.9726
]

val_losses = [
    4.1233,
    3.1149,
    2.8846,
    2.7543,
    2.7088,
    2.5594,
    2.5007,
    2.4480,
    2.3954,
    2.3902,
    2.3269,
    2.2801,
    2.2714,
    2.2662,
    2.2110,
    2.2265,
    2.1948,
    2.2111,
    2.1956,
    2.1846
]

results = {
    "epochs": 20,
    "batch_size": 64,
    "d_model": 256,
    "num_heads": 4,
    "encoder_layers": 3,
    "decoder_layers": 3,
    "feed_forward": 1024,
    "dropout": 0.1,
    "label_smoothing": 0.1,
    "optimizer": "Adam",
    "train_samples": 56336,
    "dev_samples": 8421,
    "test_samples": 15878,
    "best_epoch": 20,
    "best_validation_loss": 2.1846,
    "execution_accuracy": 0.14686988285678296,
    "logical_form_accuracy": 0.07828441869253054,
    "train_losses": train_losses,
    "validation_losses": val_losses
}

with open(
    "results.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        results,
        f,
        indent=4
    )

print("Results saved to results.json")