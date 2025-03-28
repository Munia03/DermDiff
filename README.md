## DermDiff: Generative Diffusion Model for Mitigating Racial Biases in Dermatology Diagnosis

### [Paper](https://arxiv.org/abs/2503.17536)

## Acknowledgments

This work builds upon the repository diffusers(https://github.com/huggingface/diffusers) by Huggingface.
The setup and training workflow are also adapted from the original repository.



## Stable Diffusion text-to-image fine-tuning

The `train_text_to_image.py` script shows how to fine-tune stable diffusion model on your own dataset.

___Note___:


## Running locally with PyTorch
### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

Then cd in the example folder  and run
```bash
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```



You need to accept the model license before downloading or using the weights. In this example we'll use model version `v1-4`, so you'll need to visit [its card](https://huggingface.co/CompVis/stable-diffusion-v1-4), read the license and tick the checkbox if you agree.

You have to be a registered user in 🤗 Hugging Face Hub, and you'll also need to use an access token for the code to work. For more information on access tokens, please refer to [this section of the documentation](https://huggingface.co/docs/hub/security-tokens).

Run the following command to authenticate your token

```bash
huggingface-cli login
```

If you have already cloned the repo, then you won't need to go through these steps.

<br>

#### Hardware
With `gradient_checkpointing` and `mixed_precision` it should be possible to fine tune the model on a single 24GB GPU. For higher `batch_size` and faster training it's better to use GPUs with >30GB memory.

<!-- accelerate_snippet_start -->
```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="path_to_your_dataset"

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-dermdiff-model"
```
<!-- accelerate_snippet_end -->


To run on your own training files prepare the dataset according to the format required by `datasets`, you can find the instructions for how to do that in this [document](https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder-with-metadata).
If you wish to use custom loading logic, you should modify the script, we have left pointers for that in the training script.


Once the training is finished the model will be saved in the `output_dir` specified in the command. In this example it's `sd-dermdiff-model`. To load the fine-tuned model for inference just pass that path to `StableDiffusionPipeline`

```python
import torch
from diffusers import StableDiffusionPipeline

model_path = "path_to_saved_model"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

prompt = "benign skin disease on brown skin tone"

image = pipe(prompt).images[0]
image.save("yoda-naruto.png")
```


## BibTeX

```bibtex
@article{munia2025dermdiff,
  title={DermDiff: Generative Diffusion Model for Mitigating Racial Biases in Dermatology Diagnosis},
  author={Munia, Nusrat and Imran, Abdullah-Al-Zubaer},
  journal={arXiv preprint arXiv:2503.17536},
  year={2025}
}
```




