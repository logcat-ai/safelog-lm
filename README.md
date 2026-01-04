# SafeLog-LM

A fine-tuned Gemma 3 270M model for redacting PII from Android system logs.

🤗 **Model:** [logcat-ai/safelog-lm](https://huggingface.co/logcat-ai/safelog-lm)

## What it does

SafeLog-LM automatically detects and redacts Personally Identifiable Information from Android logs (logcat, bugreports, dmesg), replacing sensitive data with category tags.
```
Input:  D/Telephony: Sending SMS to +917514605529
Output: D/Telephony: Sending SMS to [PHONE_NUMBER]

Input:  W/Auth: Bearer token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9
Output: W/Auth: Bearer token=[ACCESS_TOKEN]

Input:  D/WifiService: Connected to SSID "Varun's Home 5G" with MAC aa:bb:cc:dd:ee:ff
Output: D/WifiService: Connected to SSID "[WIFI_SSID]" with MAC [MAC_ADDRESS]
```

## Supported PII Types

| Category | Tags |
|----------|------|
| Device IDs | `[IMEI]`, `[SERIAL_NUMBER]`, `[ANDROID_ID]`, `[ADVERTISING_ID]`, `[MAC_ADDRESS]`, `[ICCID]`, `[IMSI]` |
| Network/Location | `[IP_ADDRESS]`, `[GPS_COORDINATES]`, `[WIFI_SSID]`, `[BLUETOOTH_NAME]` |
| Personal Info | `[PERSON_NAME]`, `[EMAIL]`, `[PHONE_NUMBER]`, `[ACCOUNT_NAME]` |
| Paths | `[PATH_USERNAME]` |
| Secrets | `[ACCESS_TOKEN]`, `[CERTIFICATE_FINGERPRINT]`, `[URL_WITH_PII]` |

## Installation
```bash
pip install torch transformers tqdm
```

Clone this repo:
```bash
git clone https://github.com/logcat-ai/safelog-lm.git
cd safelog-lm
```

## Usage

### Command Line
```bash
# Basic usage
python safelog.py bugreport.txt bugreport_redacted.txt

# With larger batch size (faster, needs more VRAM)
python safelog.py logcat.log logcat_clean.log --batch-size 16

# Fast mode with pre-filtering (recommended for large files)
python safelog_fast.py bugreport.txt bugreport_redacted.txt

# CPU-only (slower but no GPU needed)
python safelog.py dmesg.txt dmesg_safe.txt --cpu
```

### Python API
```python
from safelog import SafeLog

# Initialize
sl = SafeLog()

# Redact a single line
result = sl.redact("I/TelephonyManager: getDeviceId() returning 358673091234567")
print(result)
# I/TelephonyManager: getDeviceId() returning [IMEI]

# Redact an entire file
sl.redact_file("bugreport.txt", "bugreport_redacted.txt")
```

### Fast Mode (Recommended for large files)

For large bugreports (50k+ lines), use the hybrid approach which pre-filters lines using regex before running the model:
```python
from safelog_fast import SafeLogFast

sl = SafeLogFast(batch_size=16)
sl.redact_file("bugreport.txt", "bugreport_redacted.txt")
```

This typically processes only 5-15% of lines through the model, giving a 5-10x speedup.

The hybrid approach uses regex as a fast pre-filter to identify candidate lines (5-15% of a typical bugreport). The model then precisely identifies and redacts the actual PII within those lines. Regex alone can't distinguish between an IMEI and a random number, or know that "Varun's Pixel Buds" is a Bluetooth device name—that requires language understanding.

## Performance

| File Size | Lines | Mode | Time (RTX 4050) |
|-----------|-------|------|-----------------|
| Small | 1,000 | Standard | ~1 min |
| Medium | 10,000 | Standard | ~10 min |
| Large | 50,000 | Fast (hybrid) | ~5 min |
| Large | 50,000 | Standard | ~50 min |

## Model Details

- **Base Model:** google/gemma-3-270m-it
- **Parameters:** 270M
- **Training:** QLoRA fine-tuning with completion-only loss masking
- **Dataset:** ~1,770 examples (Android logs + synthetic data)
- **Accuracy:** 99.7% token accuracy on eval set

## Requirements

- Python 3.10+
- PyTorch 2.0+
- GPU with 6GB+ VRAM (or CPU with --cpu flag)

## License

This model inherits the [Gemma license](https://ai.google.dev/gemma/terms).

## Author

Built by [logcat.ai](https://logcat.ai) — AI-powered observability for Android and Linux systems.
