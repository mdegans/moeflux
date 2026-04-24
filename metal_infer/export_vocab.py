#!/usr/bin/env python3
"""Export a decoder-side vocab.bin from a HuggingFace tokenizer.json.

This is the format that infer.m's `load_vocab` reads (used only for
token-id → string decoding in log output; does not carry merges).
The companion `export_tokenizer.py` script produces the BPET format
used by tokenizer.h for *encoding* text → tokens. The C inference
engine needs both files: tokenizer.bin (BPET) and vocab.bin (this).

Binary format (little-endian):
    uint32 num_entries
    uint32 max_id
    for each entry in order of token_id 0..num_entries-1:
        uint16 byte_len
        byte[byte_len]  (UTF-8 bytes of the token string, no terminator)

If a token_id is unused (gap in the vocab), byte_len is written as 0
and no bytes follow. That way the reader can walk the file purely
sequentially without an index — matching load_vocab's sequential
fread loop.

Usage:
    python export_vocab.py <tokenizer.json> <vocab.bin>
"""

import json
import struct
import sys


def _unicode_to_byte():
    """Return a {codepoint: byte} dict inverting GPT-2's bytes_to_unicode.

    Qwen's tokenizer.json stores token strings in a "visible unicode"
    encoding: printable ASCII + Latin-1-supplement bytes keep their
    own codepoint, and the other ~68 bytes are shifted into the
    256..0x143 range so every byte maps to exactly one printable
    codepoint. This function builds the inverse.
    """
    # Bytes that keep their own codepoint.
    keep = (list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1)))
    mapping = {b: b for b in keep}
    n = 0
    for b in range(256):
        if b not in mapping:
            mapping[256 + n] = b
            n += 1
    # Invert: {codepoint: byte}. Flip keep and the shifted range.
    inv = {}
    for b in keep:
        inv[b] = b
    n = 0
    for b in range(256):
        if b not in keep:
            inv[256 + n] = b
            n += 1
    return inv


_UNICODE_TO_BYTE = _unicode_to_byte()


def decode_qwen_token(s):
    """Convert a tokenizer.json vocab key back to its raw UTF-8 bytes."""
    out = bytearray()
    for ch in s:
        c = ord(ch)
        if c in _UNICODE_TO_BYTE:
            out.append(_UNICODE_TO_BYTE[c])
        else:
            # Fallback for tokens that aren't pure byte-level BPE
            # (e.g. tokenizer that interleaves raw UTF-8) — encode
            # the codepoint directly.
            out.extend(ch.encode("utf-8"))
    return bytes(out)


def main():
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <tokenizer.json> <vocab.bin>")
    tok_path, out_path = sys.argv[1], sys.argv[2]

    with open(tok_path, "r", encoding="utf-8") as f:
        t = json.load(f)

    vocab = t["model"]["vocab"]           # {text: id}
    added = t.get("added_tokens", [])      # [{id, content, ...}]

    id_to_bytes = {}
    for text, idx in vocab.items():
        id_to_bytes[idx] = decode_qwen_token(text)
    # Added tokens (<|im_start|>, etc.) are stored literally — they
    # are not byte-level BPE encoded. Use content as-is.
    for entry in added:
        id_to_bytes[entry["id"]] = entry["content"].encode("utf-8")

    max_id = max(id_to_bytes) if id_to_bytes else 0
    num_entries = max_id + 1

    with open(out_path, "wb") as f:
        f.write(struct.pack("<II", num_entries, max_id))
        for i in range(num_entries):
            b = id_to_bytes.get(i, b"")
            if len(b) > 0xFFFF:
                sys.exit(f"error: token {i} exceeds uint16 length")
            f.write(struct.pack("<H", len(b)))
            if b:
                f.write(b)

    print(f"wrote {out_path}: num_entries={num_entries} max_id={max_id} "
          f"(gaps: {num_entries - len(id_to_bytes)})")


if __name__ == "__main__":
    main()
