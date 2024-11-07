# LLaVA Model Folder

## Apply_delta.py

Loads the base llama Model
If model alr patched (has projectors), quit
Else add weights to the delta's model dict (Delta is the weights of the model with the original llama weights taken out)
Save the new model

## Consolidate.py

Runs auto upgrade on the model and gets a checkpoint

### Utils.py

Just changes the config.....

## Builder.py

### load_pretrained_model

- Quantises it using bnb if needed, and uses correct dtype
- If multimodal, set flag and pop
- Else load normal llama
- If name is llava (meaning alr multimodal) or multimodal flag -> Check if lora available, and load it
- Add dummy parameters to to match head dim and seq len
- If the lora isnt from known sources, load from hf hub and load lora non-trainables and merge lora
- Else check if only mm projector is given and load base model

- Next, add image tokens and start and end tokens for image (We need add for audio)
- Get vision and audio towers
- Add metadata like ctx length etc and return

## llava_arch.py

### class LLavaMetaModel

- Init vision related stuff
- If FSDP then init differently
- add faster token if not alr in config
- FSDP wrapped in a list so that self reference points to FSDP object, and because vision model is a module yet also an attribute of another model, so it gets sharded too but the sharded model is not reflected without the wrapping of the module in a mutable list
- Create special tokens to deal with slowfast approach and also variable image size (unpadded image newline token)

### class LLavaMetaForCasualLM(ABC)

- get_2dpool -> Init the 2dpooling functions for slowfast (Aggregator)
- encode_images-> Encode images function to pass through encoder
- encode_multimodal -> Encode multimodal encodes the images, and then does the appropriate pooling depending on the stream. It also finds out based on the video_idx which tokens represent video tokens
  - This is our aggregator from the slowfast llava paper
- add token per grid-> for each grid flatten into a sequence with newline tokens for each row of pixels
- add_token_per_frame -> Just do channels first and add newline token

#### prepare_inputs_for_multimodal

- Three branches
----> If no vision and audio, return
----> If only audio, encode the audio, take note of positions of audio and continue
----> If vision:
      ---->
