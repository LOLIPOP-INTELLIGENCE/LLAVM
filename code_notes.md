# LLaVA Model Folder

## Apply_delta.py

Loads the base llama Model
If model alr patched (has projectors), quit
Else add weights to the delta's model dict (Delta is the weights of the model with the original llama weights taken out)
Save the new model

## Consolidate.py

Runs auto upgrade on the model and gets a checkpoint

### Utils.py

Auto upgrade -> Just changes the config.....

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

- ABC inherits (Abstract Base Class)

#### Options for vision module

Patch Merge Types:

- flat -> Takes all image patches and flatten them into a single sequence (4x4 grid -> 16 length seq)
- spatial -> Preserves 2D spatial relationships between patches (4x4 grid -> 4x4 length seq with new line tokens in btwn)
- unpad -> Removes padding that was added to make images square

```
[Pad Pad Pad]     [P1 P2]
[P1  P2  Pad]  →  [P3 P4]
[P3  P4  Pad]
```

- maxpool2x2
Applies max pooling to reduce spatial dimensions by half (4x4 grid -> 2x2 grid)

Image Aspect Ratio Options:

- square: Force images to be square
- anyres: Handles images of any res and preserves aspect ratio
- anyres_max_*: Like anyres but with max limit on patches

Token Position Options:

- grid: Adds newline token based on 2D grid location
- frame: Add newline tokens per frame
- one_token: Add single newline token for entire image/video
- no_token:No additional newline tokens

#### Functions

- get_2dpool -> Init the 2dpooling functions for slowfast (Aggregator)
- encode_images-> Encode images function to pass through encoder
- encode_multimodal -> Encode multimodal encodes the images, and then does the appropriate pooling depending on the stream. It also finds out based on the video_idx which tokens represent video tokens
  - This is our aggregator from the slowfast llava paper
- add token per grid-> for each grid flatten into a sequence with newline tokens for each row of pixels
- add_token_per_frame -> Just do channels first and add newline token
- initialize_vision_tokenizer -> Tokenize images, and start_img and end_img tokens

##### prepare_inputs_for_multimodal

- Three branches
----> If no vision and audio, return
----> If only audio, encode the audio, take note of positions of audio and continue
----> If vision:
      ----> Turn modalities into a list
      ----> Processes both single images and videos to have consistent size
            - Example:
              # Example input images list might look like:
                  images = [
                      torch.tensor(...),  # shape: (3, 224, 224)     - needs unsqueeze
                      torch.tensor(...),  # shape: (1, 3, 224, 224)  - already correct
                      torch.tensor(...),  # shape: (3, 224, 224)     - needs unsqueeze
                  ]
                  # After the list comprehension:
                  images = [
                      torch.tensor(...),  # shape: (1, 3, 224, 224)  - unsqueezed
                      torch.tensor(...),  # shape: (1, 3, 224, 224)  - unchanged
                      torch.tensor(...),  # shape: (1, 3, 224, 224)  - unsqueezed
                  ]
      -----> Track indices of video tokens
      -----> Do final formatting of image and video tokens
                Example:
                    Input could be:
                        images = [
                            tensor(1, 3, 224, 224),     # 4D: batched image
                            tensor(8, 3, 224, 224),     # 4D: video frames
                            tensor(5, 1, 3, 224, 224)   # 5D: batch of videos
                        ]
                    After processing
                        images_list = [
                            tensor(1, 3, 224, 224),     # Unchanged
                            tensor(8, 3, 224, 224),     # Unchanged
                            tensor(5, 1, 3, 224, 224)   # Unsqueezed to add batch dim
                        ]
      -----> Concat all images for batch processing
      -----> Then, take note of how many batches per image/video
      -----> Encode all features through visual encoder, then split back to images and videos (Note: If using faster video then change to self.encode_multimodals)
      -----> Using video_idx, append video frames and apply pooling to them
      -----> Normally append image features to image_feature list

      -----> Get merge type, aspect ratio and newline positions, which were described above
      -----> If flat, flatten all image features
      -----> If spatial, create list of new image features:
        1)------> if current image is a video frame, 
          ------> Check if newline position is grid
            ------> If so, add newline tokens per grid
            ------> Check if add_faster_video is True
              -----> Add faster video stream to grida
              -----> When current index % stride of fast stream, add a fast token to every frame
                1) Keeps track of temporal position
                2) Process diff quality frames appropriately
                3) Maintian seq alignment
              --> Concatenate all images into a batch
              --> This is one image feature (Representing one whole video)
          -----> If no token, just flatten
          -----> If frame, add newline token per frame
          -----> If one token, add one token per video
        2)-----> If current image is a multiple images/ multi patches
          ---> First patch is treated as base feature
          ---> image_feature are rest of the features
          ---> height and width are collected from what vision tower expects
          ---> Assert height and width are same as base image
            ----> If anyres_max, find max num patches
            ----> If anyres or anyres max, then get image shape and try to get each patch's dimensions
            ----> If square, then just get the image shape and divide by patch size
            if maxpool2x2, then apply maxpooling to reduce spatial dimensions by half
            if unpad, then remove padding that was added to make images square
            if unpad and anyres max, you also need to scale down after unpadding to get priginal shape/ interpolate for negligible scaled images
            also add newline token if unpad: Because variable width so we lose track of row boundaries (solved with unpad)
            else if nothing just flatten
            if nobase, ignore base_image feature
          Append whatever result we got as an image feature
        3)
          If unpad then unpad it and add newline token
          else just add image newline token and append image
      ####### Creating labels
      Image start and end tokens not implemented here to support pretraining
      Get labels, position ids and attn mask
      If None make dummy tensors
      Find where image, audio tokens are
        If none then just embed tokens and increment current indices by 1
        If not append the areas where there are no image/audio tokens and add to embeddings
        Embed them
        Then loop through images and audio to find multimodal features and add to embeddings
        We append cur_image_features[0:0]/audio[0:0] to maintain tensor shape -> Even if no image or audio, we shld have same shape
        so if only images instead of [text][image]
        we shld have [text][image][empty_audio] and vice versa

        ####### Tokenizing
        1) Truncate sentences which are too long, applies to both input and labels
        2) Create padding
        3) Support both left and right padding while maintaining proper seq ordering -> affects pos ID and attn masks -> only real content is attended to
        4) Stack padded tensors together
        5) can add random offsets to positions for training to learn position invariant features
      

        
