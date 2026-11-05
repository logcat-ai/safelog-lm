#!/usr/bin/env python3
"""
SafeLog Fast - PII redaction with pre-filtering for large files
Only runs the model on lines that likely contain PII.
"""

import sys
import re
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from pathlib import Path

MODEL_ID = "logcat-ai/safelog-lm"

SYSTEM_PROMPT = (
    "You are a PII redaction tool. Replace all personally identifiable information "
    "in the input text with the appropriate tag. Preserve all other text exactly. "
    "Tags: [IMEI], [SERIAL_NUMBER], [ANDROID_ID], [ADVERTISING_ID], [MAC_ADDRESS], "
    "[ICCID], [IMSI], [IP_ADDRESS], [GPS_COORDINATES], [WIFI_SSID], [BLUETOOTH_NAME], "
    "[PERSON_NAME], [EMAIL], [PHONE_NUMBER], [ACCOUNT_NAME], [PATH_USERNAME], "
    "[ACCESS_TOKEN], [CERTIFICATE_FINGERPRINT], [URL_WITH_PII]"
)

# Patterns that suggest a line might contain PII
PII_INDICATORS = re.compile(r"|".join([
    r"@",                                     # Email
    r"\b\d{10,15}\b",                      # IMEI, phone, ICCID
    r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",  # IP address
    r"([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}",   # MAC address
    r"token",                                 # Tokens
    r"bearer",
    r"auth",
    r"key=",
    r"password",
    r"secret",
    r"credential",
    r"ssid",
    r"bluetooth|btgatt|\bbt_|a2dp|\bhfp\b",     # Bluetooth device names ("Varun's Pixel Buds")
    r"device_?name|display_?name|account_?name|owner_?name|user_?name",  # Name fields
    r"imei",
    r"imsi",
    r"iccid",
    r"android_id",
    r"serial",
    r"/home/",                                # Path with username
    r"/users/",
    r"latitude|longitude|lat=|lng=|lon=",    # GPS
    r"\+\d{1,3}[-.\s]?\d",                # Phone numbers
]), re.IGNORECASE)


class SafeLogFast:
    def __init__(self, batch_size=16):
        self.batch_size = batch_size
        
        print("Loading SafeLog model...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.model.eval()
        print("Model ready!")

    def _might_contain_pii(self, line: str) -> bool:
        return bool(PII_INDICATORS.search(line))

    def _make_prompt(self, text: str) -> str:
        return f"""<start_of_turn>user
{SYSTEM_PROMPT}

Input: {text}<end_of_turn>
<start_of_turn>model
"""

    @torch.inference_mode()
    def redact_batch(self, lines: list[str]) -> list[str]:
        if not lines:
            return lines
        
        prompts = [self._make_prompt(l) for l in lines]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        results = []
        for output in outputs:
            response = self.tokenizer.decode(
                output[inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            results.append(response.strip())
        
        return results

    def redact_file(self, input_path: str, output_path: str):
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        with open(input_path, "r", errors="ignore") as f:
            lines = f.readlines()
        
        total = len(lines)
        
        # Pre-filter
        pii_candidates = []
        for i, line in enumerate(lines):
            if self._might_contain_pii(line):
                pii_candidates.append((i, line.rstrip("\n\r")))
        
        print(f"Total lines: {total:,}")
        print(f"Lines with potential PII: {len(pii_candidates):,} ({100*len(pii_candidates)/total:.1f}%)")
        print(f"Lines skipped: {total - len(pii_candidates):,}")
        
        results = list(lines)
        
        if pii_candidates:
            with tqdm(total=len(pii_candidates), desc="Redacting", unit="lines") as pbar:
                for i in range(0, len(pii_candidates), self.batch_size):
                    batch = pii_candidates[i:i + self.batch_size]
                    indices, texts = zip(*batch)
                    
                    redacted = self.redact_batch(texts)
                    
                    for idx, text in zip(indices, redacted):
                        results[idx] = text + "\n"
                    
                    pbar.update(len(batch))
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.writelines(results)
        
        print(f"Done! Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SafeLog Fast - PII redaction with pre-filtering")
    parser.add_argument("input", help="Input log file")
    parser.add_argument("output", help="Output file path")
    parser.add_argument("--batch-size", "-b", type=int, default=16)
    
    args = parser.parse_args()
    
    try:
        safelog = SafeLogFast(batch_size=args.batch_size)
        safelog.redact_file(args.input, args.output)
    except torch.cuda.OutOfMemoryError:
        print("\nOut of GPU memory! Try: --batch-size 8")
        sys.exit(1)


if __name__ == "__main__":
    main()
