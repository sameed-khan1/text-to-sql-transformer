import json
import math
import argparse
import os
import torch
import torch.nn as nn
import sentencepiece as spm
import matplotlib.pyplot as plt

from dataset import make_loader
from transformer import TransformerModel
from tokenizer import PAD_ID


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE   = 64
D_MODEL      = 256
WARMUP_STEPS = 4000
CHECKPOINT   = "best_model.pt"


# ---------------------------------------------------------------------------
# Noam learning-rate schedule (Vaswani et al., Eq. 3)
# lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
# ---------------------------------------------------------------------------

def noam_lr(step, d_model=D_MODEL, warmup=WARMUP_STEPS):
    """Returns the LR factor at a given (1-indexed) global step."""
    step = max(step, 1)
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))


# ---------------------------------------------------------------------------
# Per-epoch helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, criterion, step_counter):
    model.train()
    total_loss = 0.0

    for src, tgt in loader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        decoder_input = tgt[:, :-1]
        target        = tgt[:, 1:]

        optimizer.zero_grad()
        logits = model(src, decoder_input)
        loss   = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        step_counter[0] += 1

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            decoder_input = tgt[:, :-1]
            target        = tgt[:, 1:]
            logits = model(src, decoder_input)
            loss   = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def is_old_checkpoint(path):
    """
    Returns True if the checkpoint was saved without the Noam schedule
    (i.e. missing the 'step' key — these were saved by the old flat-LR train.py).
    """
    if not os.path.exists(path):
        return False
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        return "step" not in ck
    except Exception:
        return False


def load_checkpoint(path, model, optimizer=None):
    """Load checkpoint. Returns (start_epoch, global_step, best_val_loss)."""
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    start_epoch   = ck.get("epoch", 0) + 1
    global_step   = ck.get("step",  0)
    best_val_loss = ck.get("val_loss", math.inf)
    print(f"  Loaded checkpoint from epoch {ck.get('epoch','?')}, "
          f"global step {global_step}, val_loss {best_val_loss:.4f}")
    return start_epoch, global_step, best_val_loss


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--scratch", action="store_true",
                    help="Retrain from random init (ignores existing checkpoint)")
parser.add_argument("--resume",  action="store_true",
                    help="Force resume from best_model.pt")
parser.add_argument("--epochs",  type=int, default=20,
                    help="Total epochs to train (default: 20)")
args = parser.parse_args()

EPOCHS = args.epochs

# ---------------------------------------------------------------------------
# Decide: scratch or resume?
# ---------------------------------------------------------------------------

old_ck = is_old_checkpoint(CHECKPOINT)
ck_exists = os.path.exists(CHECKPOINT)

if args.scratch:
    mode = "scratch"
elif args.resume and ck_exists:
    mode = "resume"
elif ck_exists and not old_ck:
    # New-style checkpoint exists → resume automatically
    mode = "resume"
elif ck_exists and old_ck:
    # Old flat-LR checkpoint detected → warn and retrain from scratch
    print("=" * 60)
    print("WARNING: existing best_model.pt was trained with a FLAT learning rate")
    print("         (old training code — no 'step' key in checkpoint).")
    print("         Retraining from scratch with the correct Noam schedule.")
    print("         Use --resume to force-load old weights anyway.")
    print("=" * 60)
    mode = "scratch"
else:
    mode = "scratch"

print(f"\nMode: {mode.upper()}  |  Epochs: {EPOCHS}  |  Device: {DEVICE}")

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

sp = spm.SentencePieceProcessor(model_file="sql_sp.model")

train_loader = make_loader("train_pairs.jsonl", sp, train=True,  batch_size=BATCH_SIZE)
dev_loader   = make_loader("dev_pairs.jsonl",   sp, train=False, batch_size=BATCH_SIZE)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = TransformerModel(
    vocab_size=sp.get_piece_size(),
    d_model=D_MODEL,
    num_heads=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    d_ff=1024,
    dropout=0.1,
    pad_id=PAD_ID,
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")

assert model.output.weight is model.embedding.weight, \
    "Weight sharing broken!"
print("Weight sharing: OK")

# ---------------------------------------------------------------------------
# Loss + optimiser + Noam scheduler
# ---------------------------------------------------------------------------

criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1.0,
    betas=(0.9, 0.98),
    eps=1e-9,
)

# Global step counter — used for Noam schedule.
# If resuming, we start the counter at the saved step so the LR
# continues its decay from where training left off.
step_counter = [0]

def lr_lambda(_):
    # scheduler.step() is called after every batch update.
    # step_counter is incremented inside train_epoch, so we read
    # the current value here (already incremented).
    return noam_lr(step_counter[0])

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# ---------------------------------------------------------------------------
# Load checkpoint if resuming
# ---------------------------------------------------------------------------

start_epoch   = 1
best_val_loss = math.inf
best_epoch    = 1
train_losses  = []
val_losses    = []

if mode == "resume":
    start_epoch, step_counter[0], best_val_loss = load_checkpoint(
        CHECKPOINT, model
    )
    best_epoch = start_epoch - 1
    # Fast-forward scheduler to the saved step without updating weights
    # (just to set the internal state correctly)
    for _ in range(step_counter[0]):
        scheduler.step()
    remaining = EPOCHS - (start_epoch - 1)
    print(f"Resuming from epoch {start_epoch}  "
          f"({remaining} epoch(s) remaining, "
          f"LR now: {optimizer.param_groups[0]['lr']:.2e})")
else:
    print("Starting fresh training from random initialisation.")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)

for epoch in range(start_epoch, EPOCHS + 1):

    train_loss = train_epoch(
        model, train_loader, optimizer, scheduler, criterion, step_counter
    )
    val_loss = evaluate(model, dev_loader, criterion)
    current_lr = optimizer.param_groups[0]["lr"]

    train_losses.append(round(train_loss, 4))
    val_losses.append(round(val_loss, 4))

    print(
        f"Epoch {epoch:02d}/{EPOCHS}  "
        f"Train Loss: {train_loss:.4f}  "
        f"Val Loss: {val_loss:.4f}  "
        f"LR: {current_lr:.2e}  "
        f"Step: {step_counter[0]}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch    = epoch

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch":     epoch,
                "val_loss":  val_loss,
                "step":      step_counter[0],    # ← required for resume
            },
            CHECKPOINT,
        )
        print(f"  → Best model saved  (val_loss={val_loss:.4f})")

print("\nTraining complete.")
print(f"Best epoch: {best_epoch}  |  Best val loss: {best_val_loss:.4f}")

# ---------------------------------------------------------------------------
# Save results.json
# ---------------------------------------------------------------------------

# Load existing results.json to preserve accuracy fields if they exist
results_existing = {}
if os.path.exists("results.json"):
    try:
        with open("results.json", encoding="utf-8") as f:
            results_existing = json.load(f)
    except Exception:
        pass

results = {
    "epochs":               EPOCHS,
    "best_epoch":           best_epoch,
    "best_validation_loss": round(best_val_loss, 4),
    "batch_size":           BATCH_SIZE,
    "d_model":              D_MODEL,
    "num_heads":            4,
    "encoder_layers":       3,
    "decoder_layers":       3,
    "feed_forward":         1024,
    "dropout":              0.1,
    "label_smoothing":      0.1,
    "optimizer":            "Adam (Noam schedule, warmup=4000)",
    "warmup_steps":         WARMUP_STEPS,
    "train_samples":        len(train_loader.dataset),
    "dev_samples":          len(dev_loader.dataset),
    # Preserve existing accuracy numbers if evaluate_model.py was already run
    "execution_accuracy":   results_existing.get("execution_accuracy",   0.0),
    "logical_form_accuracy":results_existing.get("logical_form_accuracy", 0.0),
    "train_losses":         train_losses,
    "validation_losses":    val_losses,
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)
print("Saved results.json")

# ---------------------------------------------------------------------------
# Plot 1: Training & Validation Loss
# ---------------------------------------------------------------------------

ep_range = range(start_epoch, start_epoch + len(train_losses))

plt.figure(figsize=(10, 5))
plt.plot(ep_range, train_losses, label="Training Loss")
plt.plot(ep_range, val_losses,   label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training and Validation Loss per Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_validation_loss.png", dpi=150)
plt.close()
print("Saved training_validation_loss.png")

# ---------------------------------------------------------------------------
# Plot 2: Learning-Rate Schedule (first 20,000 steps)
# ---------------------------------------------------------------------------

steps = range(1, 20_001)
lrs   = [noam_lr(s) for s in steps]

plt.figure(figsize=(10, 4))
plt.plot(steps, lrs)
plt.axvline(x=WARMUP_STEPS, color="red", linestyle="--",
            label=f"Warmup end ({WARMUP_STEPS} steps)")
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.title(f"Noam LR Schedule (warmup={WARMUP_STEPS}, d_model={D_MODEL})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lr_schedule.png", dpi=150)
plt.close()
print("Saved lr_schedule.png")