# Examples of exporting [SKT-AI/KoBART](https://github.com/SKT-AI/KoBART) to ONNX

## Environments

```
python==3.8.12
pytorch==1.10.0
transformers==https://github.com/swoook/transformers-cloned
onnxruntime-gpu==1.9.0
```

* `transformers<=4.13.0` currently supports exporting only `BartModel`, not variants including `BartForSequenceClassification`, `BartForQuestionAnswering`, and so on

## Instruction: Export to ONNX

1. Place files for KoBART-base and its tokenizer in same directory `$SRC_DIR`

   ```
   config.json
   emji_tokenizer-merges.txt
   pytorch_model.bin
   model.json
   emji_tokenizer-vocab.json
   ```

2. Change some filenames like:

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
   $ python ./export2onnx.py --model $SRC_DIR --feature $FEATURE $DST_DIR
   ```
   
   | arguments  | descriptions                                                 |
   | ---------- | ------------------------------------------------------------ |
   | `$SRC_DIR` | a directory of files for KoBART-base and its tokenizer       |
   | `$FEATURE` | `default` for `BartModel`<br/>`sequence-classification` for `BartforSequenceClassification` |
   | `$DST_DIR` | a directory of ONNX files                                    |

# Appendix

## [x] values not close enough

* In some cases, we may got the error message below:

```
ValueError: Outputs values doesn't match between reference model and ONNX exported model: Got max absolute difference of: $DIFF
```

* In this case, we should consider training with mixed precision

## Compatibility with `pytorch-lightning`

- Recall the examples of [$REPO_ROOT/examples in SKT-AI/KoBART (github)](https://github.com/SKT-AI/KoBART/tree/main/examples)
- They use `pytorch_lightning`
- And `pytorch_lightning` saves the model into *.ckpt* and *.yaml* file
- However, `transformers.export2onnx` does NOT support them
- We can reverse them to the format of Hugging Face:

```python
paths = dict()
paths['ckpt'] = $CKPT_PATH
paths['yaml'] = $YAML_PATH
paths['huggingface'] = $OUTPUT_DIR

# KoBARTClassification implmented in $REPO_ROOT/examples/nsmc_classification/nsmc.py
pytorch_lightning_wrapper = KoBARTClassification.load_from_checkpoint(
    checkpoint_path=paths['ckpt'],
    hparams_file=paths['yaml'],
    map_location=None,
)

pytorch_lightning_wrapper.model.save_pretrained(paths['huggingface'])
```

## When we have different files for tokenizer

* Sometimes we might have different files for tokenizer for some reasons
* We can make necessary files by:

```python
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(
tokenizer_file=$SRC_DIR,
bos_token="<s>",
eos_token="</s>",
unk_token="<unk>",
pad_token="<pad>",
mask_token="<mask>",
)
tokenizer.save_pretrained($DST_DIR)
```

