# Example: Export KoBART-base to ONNX

## Environments

```
python==3.8.12
pytorch==1.10.0
transformers==4.12.5
onnxruntime-gpu==1.9.0
```

## Export to ONNX

1. Place files for KoBART-base and its tokenizer in same directory `$SRC_DIR`

   ```
   config.json
   emji_tokenizer-merges.txt
   pytorch_model.bin
   model.json
   emji_tokenizer-vocab.json
   ```

2. Change the some filenames like:

| before                      | after            |
| --------------------------- | ---------------- |
| *emji_tokenizer-merges.txt* | *merges.txt*     |
| *model.json*                | *tokenizer.json* |
| *emji_tokenizer-vocab.json* | *vocab.json*     |

* Then, `$SRC_DIR` includes:

```
config.json
merges.txt
pytorch_model.bin
tokenizer.json
vocab.json
```

2. Execute the command below to export

   ```
   $ python ./export2onnx.py --model $SRC_DIR --feature default $DST_DIR
   ```

# Appendix

## [x] values not close enough

* In some cases, we may got the error message below:

```
ValueError: Outputs values doesn't match between reference model and ONNX exported model: Got max absolute difference of: $DIFF
```

* In this case, we should consider training with mixed precision
