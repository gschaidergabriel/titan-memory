#!/bin/bash
# Reassemble the adapter from split parts (GitHub has a 100MB file limit)
cat adapter_model.safetensors.part_* > adapter_model.safetensors
echo "Reassembled: $(wc -c < adapter_model.safetensors) bytes"
