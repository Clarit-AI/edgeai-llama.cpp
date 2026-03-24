#!/usr/bin/env python3
"""
Add thinking token aliases to a GGUF file.

This script patches a GGUF file to add aliases that map universal thinking markers
(e.g., <think>/</think>) to model-specific thinking tokens (e.g., ◁think▷/◁/think▷).

Usage:
    python add_thinking_token_alias.py <input.gguf> <output.gguf> [--think-start TOKEN] [--think-end TOKEN]

Example:
    python add_thinking_token_alias.py kimi-model.gguf kimi-model-patched.gguf \\
        --think-start "◁think▷" --think-end "◁/think▷"
"""

import argparse
import json
import sys
import os

# Add gguf-py to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'gguf-py'))
import gguf


def find_token_id(vocab, token_text):
    """Find the token ID for a given token text in the vocabulary."""
    # Handle byte-encoded tokens (e.g., "<0xXX>")
    if token_text.startswith('<0x') and token_text.endswith('>'):
        return None

    # Try direct lookup in token_to_id
    if token_text in vocab.get('token_to_id', {}):
        return vocab['token_to_id'][token_text]

    # Try reverse lookup in id_to_token
    id_to_token = vocab.get('id_to_token', {})
    for tid, text in id_to_token.items():
        if text == token_text:
            return tid

    return None


def main():
    parser = argparse.ArgumentParser(
        description='Add thinking token aliases to a GGUF file'
    )
    parser.add_argument('input', help='Input GGUF file')
    parser.add_argument('output', help='Output GGUF file')
    parser.add_argument('--think-start', default='◁think▷',
                        help='Thinking start token text (default: ◁think▷)')
    parser.add_argument('--think-end', default='◁/think▷',
                        help='Thinking end token text (default: ◁/think▷)')

    args = parser.parse_args()

    print(f"Reading {args.input}...")
    reader = gguf.GGUFReader(args.input)

    # Build token-to-id mapping from the vocab
    token_to_id = {}
    id_to_token = {}

    # Find the vocab key
    vocab_key = None
    for key in reader.fields.keys():
        if 'tokenizer.ggml.tokens' in key:
            vocab_key = key
            break

    if not vocab_key:
        print("ERROR: Could not find tokenizer tokens in GGUF file")
        sys.exit(1)

    # Get the token type key to find token texts
    token_types_key = vocab_key.replace('tokens', 'token_type')

    tokens_data = reader.fields[vocab_key]
    # Use contents() to get the actual token list
    if hasattr(tokens_data, 'contents'):
        tokens = []
        for token in tokens_data.contents():
            if isinstance(token, bytes):
                tokens.append(token.decode('utf-8', errors='replace'))
            else:
                tokens.append(str(token))
    elif hasattr(tokens_data, 'data') and hasattr(tokens_data, 'parts'):
        # Fallback: use data indices to extract real tokens from parts.
        # parts elements are numpy ndarrays, so convert via bytes() first.
        tokens = [str(bytes(tokens_data.parts[idx]), encoding='utf-8', errors='replace') for idx in tokens_data.data]
    else:
        tokens = [tokens_data]

    # Try to get token type count
    n_token_types = 0
    if token_types_key in reader.fields:
        ntypes_data = reader.fields[token_types_key]
        if hasattr(ntypes_data, 'parts'):
            n_token_types = int(ntypes_data.parts[0])
        else:
            n_token_types = int(ntypes_data)

    print(f"Found {len(tokens)} tokens, {n_token_types} token types")

    # Build mappings
    for i, token_text in enumerate(tokens):
        token_to_id[token_text] = i
        id_to_token[i] = token_text

    # Find thinking tokens
    think_start_id = find_token_id(
        {'token_to_id': token_to_id, 'id_to_token': id_to_token},
        args.think_start)
    think_end_id = find_token_id(
        {'token_to_id': token_to_id, 'id_to_token': id_to_token},
        args.think_end)

    if think_start_id is None:
        print("WARNING: Could not find thinking start token '{}' in vocab".format(args.think_start))
        print("  Available tokens containing '◁':")
        for t in tokens:
            if '◁' in t:
                print("    '{}' -> {}".format(t, token_to_id.get(t, 'NOT FOUND')))
        sys.exit(1)

    if think_end_id is None:
        print("WARNING: Could not find thinking end token '{}' in vocab".format(args.think_end))
        print("  Available tokens containing '◁':")
        for t in tokens:
            if '◁' in t:
                print("    '{}' -> {}".format(t, token_to_id.get(t, 'NOT FOUND')))
        sys.exit(1)

    print(f"Found think start token '{args.think_start}' -> ID {think_start_id}")
    print(f"Found think end token '{args.think_end}' -> ID {think_end_id}")

    # Create aliases dict
    aliases = {
        "<think>": [think_start_id],
        "</think>": [think_end_id]
    }
    aliases_json = json.dumps(aliases)
    print(f"Created alias JSON: {aliases_json}")

    # Read all data from input file
    print(f"Copying to {args.output}...")
    writer = gguf.GGUFWriter(args.output, reader.arch)

    # Copy all existing fields
    for key, field in reader.fields.items():
        if key == gguf.GGUFKeys.Tokenizer.THINKING_TOKEN_ALIAS:
            # Skip existing alias key, we'll add our own
            continue
        # Extract value and type from ReaderField
        if hasattr(field, 'contents') and hasattr(field, 'types'):
            value = field.contents()
            vtype = field.types[0]
            writer.add_key_value(key, value, vtype)
        else:
            # Fallback for simple fields
            writer.add_key_value(key, field.parts[field.data[0]] if hasattr(field, 'data') else field, field.types[0])

    # Add the thinking token alias
    writer.add_string(gguf.GGUFKeys.Tokenizer.THINKING_TOKEN_ALIAS, aliases_json)

    # Copy tensor data
    print("Copying tensor data...")
    for tensor in reader.tensors:
        writer.write_tensor(tensor.name, tensor.data, tensor.tensor_type)

    writer.close()
    print(f"Done! Patched GGUF saved to {args.output}")


if __name__ == '__main__':
    main()