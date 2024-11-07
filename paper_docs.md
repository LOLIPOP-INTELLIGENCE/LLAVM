# LLAVA-NEXT

## High Level Gist of Paper

### Video Representation for LLMs

- LLAVA considers that the number of visual tokens is crucial for preserving informational content of each frame, vital to video comprehension.
- Video frames are categorized into two groups based on strike rate s
- s frames are uniformly selected to form slow frame group, and the rest are fast frames
- Then apply different pooling rates on the different frames

# Slow Fast LLAVA (Inspiration for LLAVA-NEXT)

## Slow fast idea

- Slow pathway extracts features at a low frame rate while keeping as much spatial detail as possible. (Less frames, more visual info preserved)
- Fast pathway operates on a high frame rate but uses larger spatial pooling stride (More frames, less visual info preserved)

### Why this idea

=> Originally high computational and labelling cost to train a model with high frame rate
-> Main drawbacks are due to cost, limited no of frame rates, hard to capture temporal and spatila content
-> Feed video features into LLM without proper temporal modeling design, rely on LLM that it will hopefully model motion patterns

### Slow Fast in detail

Video LLM order:

1) Frame sampler selects N key frames -> Arranged as a combined image grid or treated independently
2) Extracted into a Visual Encoder -> features F_y
3) Before inputting features into LLM, we have aggregator to aggregate visual features

- leverage temporal prior knowledge for better video repr. and  reduce no of video tokens
    -> This is where the two streams comes in

4) Both aggregated video features and sys prompt and question is then fed into LLM

#### Pathway details

- Slow pathway samples N_slow frames , operates o less frames but does a spatial pooling and temporal downsample
--> He says properly pooling but what???
- Fast pathway just does a spatial pool, used to capture as much temporal context as possible in a frame
- Final features are F_aggr = [flat(F_slow), flat(F_fast)]
- We then concatenate these features with the text tokens
- (They don't use any tokens for separating fast and slow pathways, but there is one in LLAVA-NEXT (Question))
