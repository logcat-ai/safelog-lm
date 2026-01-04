#!/usr/bin/env python3
"""
SafeLog - Redact PII from Android system logs
Model: logcat-ai/safelog-lm

Usage:
    python safelog.py <input_file> <output_file> [--batch-size 4]
"""

import sys
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


class SafeLog:
    def __init__(self, batch_size=4, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.batch_size = batch_size
        
        print(f"Loading SafeLog model on {device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device if device == "cuda" else None,
        )
        
        if device == "cpu":
            self.model = self.model.to(device)
        
        self.model.eval()
        print("Model ready!")

    def _make_prompt(self, text: str) -> str:
        return f"""<start_of_turn>user
{SYSTEM_PROMPT}

Input: {text}<end_of_turn>
<start_of_turn>model
"""

    def redact(self, text: str) -> str:
        """Redact PII from a single line of text."""
        if len(text.strip()) < 3:
            return text
        
        prompt = self._make_prompt(text.strip())
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()

    @torch.inference_mode()
    def redact_batch(self, lines: list[str]) -> list[str]:
        """Redact PII from a batch of lines."""
        valid = [(i, line.strip()) for i, line in enumerate(lines) if len(line.strip()) >= 3]
        
        if not valid:
            return lines
        
        indices, texts = zip(*valid)
        prompts = [self._make_prompt(t) for t in texts]
        
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        results = list(lines)
        for idx, output in zip(indices, outputs):
            response = self.tokenizer.decode(
                output[inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            results[idx] = response.strip()
        
        return results

    def redact_file(self, input_path: str, output_path: str, show_progress=True):
        """Redact PII from an entire file."""
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        with open(input_path, "r", errors="ignore") as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        print(f"Processing {total_lines:,} lines from {input_path.name}")
        
        redacted_lines = []
        iterator = range(0, total_lines, self.batch_size)
        
        if show_progress:
            iterator = tqdm(iterator, desc="Redacting", unit="batch")
        
        for i in iterator:
            batch = lines[i:i + self.batch_size]
            batch_stripped = [line.rstrip("\n\r") for line in batch]
            redacted = self.redact_batch(batch_stripped)
            redacted_lines.extend([line + "\n" for line in redacted])
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.writelines(redacted_lines)
        
        print(f"Done! Saved to {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="SafeLog - Redact PII from Android system logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python safelog.py bugreport.txt bugreport_redacted.txt
  python safelog.py logcat.log logcat_clean.log --batch-size 8
  python safelog.py dmesg.txt dmesg_safe.txt --cpu
        """
    )
    
    parser.add_argument("input", help="Input log file")
    parser.add_argument("output", help="Output file path")
    parser.add_argument("--batch-size", "-b", type=int, default=4, 
                        help="Batch size (default: 4)")
    parser.add_argument("--cpu", action="store_true", 
                        help="Force CPU inference")
    
    args = parser.parse_args()
    
    device = "cpu" if args.cpu else None
    
    try:
        safelog = SafeLog(batch_size=args.batch_size, device=device)
        safelog.redact_file(args.input, args.output)
    except torch.cuda.OutOfMemoryError:
        print("\nOut of GPU memory! Try: --batch-size 2")
        sys.exit(1)


if __name__ == "__main__":
    main()
